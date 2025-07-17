import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('US-pumpkins.csv')


# 数据预处理
def preprocess_data(df):
    # 处理日期列
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # 计算平均价格作为目标变量
    df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2

    # 选择有用的特征
    features = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Month', 'Year']
    target = 'Avg_Price'

    # 处理分类变量
    categorical_cols = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 删除缺失值
    df = df.dropna(subset=features + [target])

    return df[features + [target]]


# 预处理数据
processed_df = preprocess_data(df.copy())

# 划分特征和目标变量
X = processed_df.drop('Avg_Price', axis=1)
y = processed_df['Avg_Price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征选择1: 所有特征
features1 = X.columns.tolist()

# 特征选择2: 基于相关性的特征选择
corr_matrix = processed_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('特征相关性矩阵')
plt.show()

# 选择相关性较高的特征
features2 = ['Variety', 'Origin', 'Item Size', 'Month']


# 模型训练和评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_set_name, feature_names):
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} - 特征重要性 ({feature_set_name})")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    # 可视化决策树
    if isinstance(model, DecisionTreeRegressor):
        plt.figure(figsize=(20, 10))
        plot_tree(model, filled=True, feature_names=feature_names, max_depth=2)
        plt.title(f"决策树可视化 ({feature_set_name})")
        plt.show()

    return {
        'model_name': model_name,
        'feature_set': feature_set_name,
        'cv_rmse_mean': np.mean(cv_rmse),
        'cv_rmse_std': np.std(cv_rmse),
        'test_rmse': rmse,
        'r2_score': r2
    }


# 定义模型
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# 参数网格
param_grids = {
    'Decision Tree': {'max_depth': [3, 5, 7, None]},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
    'LightGBM': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
}

# 结果存储
results = []

# 对两组特征分别训练和评估模型
for feature_set, features in [('所有特征', features1), ('选择特征', features2)]:
    X_train_fs = X_train[features]
    X_test_fs = X_test[features]

    # 标准化
    scaler_fs = StandardScaler()
    X_train_fs_scaled = scaler_fs.fit_transform(X_train_fs)
    X_test_fs_scaled = scaler_fs.transform(X_test_fs)

    for name, model in models.items():
        print(f"\n训练 {name} 使用 {feature_set}...")

        # 网格搜索调参
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_fs_scaled, y_train)

        # 最佳模型
        best_model = grid_search.best_estimator_

        # 评估
        result = evaluate_model(
            best_model,
            X_train_fs_scaled,
            y_train,
            X_test_fs_scaled,
            y_test,
            name,
            feature_set,
            features  # 传递特征名称列表
        )

        # 添加最佳参数
        result['best_params'] = grid_search.best_params_

        results.append(result)

# 结果展示
results_df = pd.DataFrame(results)
print("\n模型性能对比:")
print(results_df[['model_name', 'feature_set', 'test_rmse', 'r2_score', 'best_params']])

# 可视化模型对比
plt.figure(figsize=(12, 6))
sns.barplot(x='model_name', y='test_rmse', hue='feature_set', data=results_df)
plt.title('不同模型和特征集的测试RMSE对比')
plt.ylabel('RMSE')
plt.xlabel('模型')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='model_name', y='r2_score', hue='feature_set', data=results_df)
plt.title('不同模型和特征集的R²分数对比')
plt.ylabel('R²分数')
plt.xlabel('模型')
plt.xticks(rotation=45)
plt.show()