import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import joblib
import time
from sklearn.inspection import permutation_importance
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 设置可视化风格
sns.set_style("whitegrid")
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class PumpkinPricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = []
        self.best_model = None
        self.feature_importances = None

    def load_data(self):
        """加载数据并进行初步处理"""
        self.df = pd.read_csv(self.data_path)

        # 添加数据质量检查
        print("数据概览:")
        print(f"行数: {self.df.shape[0]}, 列数: {self.df.shape[1]}")
        print("\n缺失值统计:")
        print(self.df.isnull().sum())

    def preprocess_data(self):
        """数据预处理"""
        df = self.df.copy()

        # 1. 处理日期特征
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

        # 2. 目标变量处理
        df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2
        df['Price_Range'] = df['High Price'] - df['Low Price']

        # 3. 处理分类变量
        categorical_cols = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color']

        # 4. 处理数值型变量
        numeric_cols = ['Month', 'Year', 'Day', 'DayOfYear', 'WeekOfYear']

        # 5. 删除不必要的列和缺失值
        cols_to_drop = ['Date', 'Low Price', 'High Price', 'Market', 'Grade', 'Sub Variety',
                        'Repack', 'Trans Mode', 'Unnamed: 0', 'Origin City', 'Origin State']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        # 6. 删除目标变量为缺失的行
        df = df.dropna(subset=['Avg_Price'])

        self.processed_df = df

        # 7. 数据探索可视化
        self._exploratory_analysis()

    def _exploratory_analysis(self):
        """数据探索分析"""
        df = self.processed_df

        # 1. 价格分布
        plt.figure(figsize=(12, 6))
        sns.histplot(df['Avg_Price'], kde=True, bins=30)
        plt.title('南瓜平均价格分布')
        plt.xlabel('价格')
        plt.ylabel('频数')
        plt.show()

        # 2. 价格随时间变化
        plt.figure(figsize=(12, 6))
        df.groupby('Month')['Avg_Price'].mean().plot()
        plt.title('不同月份的平均价格变化')
        plt.xlabel('月份')
        plt.ylabel('平均价格')
        plt.show()

        # 3. 类别变量与价格的关系
        categorical_cols = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color']
        for col in categorical_cols:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y='Avg_Price', data=df)
            plt.title(f'{col}与价格的关系')
            plt.xticks(rotation=45)
            plt.show()

    def prepare_features(self):
        """准备特征和划分数据集"""
        df = self.processed_df

        # 定义特征和目标
        categorical_cols = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color']
        numeric_cols = ['Month', 'Year', 'Day', 'DayOfYear', 'WeekOfYear']

        # 创建预处理管道
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)])

        # 划分数据集
        X = df.drop(['Avg_Price', 'Price_Range'], axis=1)
        y = df['Avg_Price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # 应用预处理
        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)

        # 保存预处理对象
        joblib.dump(preprocessor, 'output/preprocessor.joblib')

    def initialize_models(self):
        """初始化模型和参数网格"""
        self.models = {
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 9, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 1.0]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            }
        }

    def train_and_evaluate(self):
        """训练和评估模型"""
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model_info in self.models.items():
            print(f"\n=== 正在训练 {model_name} ===")
            start_time = time.time()

            # 网格搜索
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1)

            grid_search.fit(self.X_train, self.y_train)

            # 获取最佳模型
            best_model = grid_search.best_estimator_

            # 评估模型
            y_pred = best_model.predict(self.X_test)

            # 计算指标
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            # 交叉验证分数
            cv_scores = cross_val_score(
                best_model, self.X_train, self.y_train,
                cv=kfold, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)

            # 保存结果
            result = {
                'model_name': model_name,
                'best_params': grid_search.best_params_,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'cv_rmse_mean': np.mean(cv_rmse),
                'cv_rmse_std': np.std(cv_rmse),
                'training_time': time.time() - start_time
            }

            self.results.append(result)

            # 保存最佳模型
            self.models[model_name]['best_model'] = best_model
            self.models[model_name]['results'] = result

            # 特征重要性
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importances = self._get_feature_importance(best_model, model_name)

            # 打印结果
            print(f"\n{model_name} 结果:")
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"测试集 RMSE: {rmse:.4f}")
            print(f"测试集 MAE: {mae:.4f}")
            print(f"测试集 R²: {r2:.4f}")
            print(f"交叉验证 RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
            print(f"训练时间: {result['training_time']:.2f} 秒")

        # 确定最佳模型
        self._select_best_model()

    def _get_feature_importance(self, model, model_name):
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # 对于树模型，可以直接获取重要性
            if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor,
                                  GradientBoostingRegressor, xgb.XGBRegressor, lgb.LGBMRegressor)):
                return importances

        # 对于没有内置重要性方法的模型，使用排列重要性
        result = permutation_importance(
            model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1)
        return result.importances_mean

    def _select_best_model(self):
        """选择表现最好的模型"""
        results_df = pd.DataFrame(self.results)
        best_idx = results_df['test_rmse'].idxmin()
        self.best_model = {
            'name': results_df.loc[best_idx, 'model_name'],
            'model': self.models[results_df.loc[best_idx, 'model_name']]['best_model'],
            'results': results_df.loc[best_idx].to_dict()
        }

        print("\n=== 最佳模型 ===")
        print(f"模型: {self.best_model['name']}")
        print(f"测试集 RMSE: {self.best_model['results']['test_rmse']:.4f}")
        print(f"测试集 R²: {self.best_model['results']['test_r2']:.4f}")

    def visualize_results(self):
        """可视化模型结果"""
        results_df = pd.DataFrame(self.results)

        # 1. 模型性能比较
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model_name', y='test_r2', data=results_df)
        plt.title('不同模型的R²分数比较')
        plt.ylabel('R²分数')
        plt.xlabel('模型')
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='model_name', y='test_rmse', data=results_df)
        plt.title('不同模型的RMSE比较')
        plt.ylabel('RMSE')
        plt.xlabel('模型')
        plt.xticks(rotation=45)
        plt.show()

        # 2. 特征重要性可视化
        for model_name, model_info in self.models.items():
            if 'best_model' in model_info and hasattr(model_info['best_model'], 'feature_importances_'):
                plt.figure(figsize=(12, 6))
                importances = model_info['best_model'].feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # 只显示前20个重要特征

                plt.title(f"{model_name} - 特征重要性")
                plt.bar(range(len(indices)), importances[indices], align="center")
                plt.xticks(range(len(indices)), indices, rotation=90)
                plt.xlim([-1, len(indices)])
                plt.tight_layout()
                plt.show()

    def save_models(self):
        """保存训练好的模型"""
        for model_name, model_info in self.models.items():
            if 'best_model' in model_info:
                joblib.dump(model_info['best_model'], f'{model_name.replace(" ", "_")}_model.joblib')
        print("所有模型已保存")

    def run_pipeline(self):
        """运行完整的数据处理和分析流程"""
        self.load_data()
        self.preprocess_data()
        self.prepare_features()
        self.initialize_models()
        self.train_and_evaluate()
        self.visualize_results()
        self.save_models()


# 使用示例
if __name__ == "__main__":
    predictor = PumpkinPricePredictor('US-pumpkins.csv')
    predictor.run_pipeline()