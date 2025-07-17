import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from matplotlib import rcParams


# 1. 全局设置
def setup_environment():
    """配置绘图环境和警告设置"""
    # 设置字体为支持中文
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')
    print("环境设置完成")


# 2. 数据加载与预处理
def load_and_preprocess_data(filepath):
    """
    加载并预处理能源数据
    参数:
        filepath: 数据文件路径
    返回:
        DataFrame: 处理后的数据
    """
    print("\n正在加载数据...")
    data = pd.read_csv(filepath, parse_dates=['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)
    print(f"数据加载完成，共 {len(data)} 条记录")
    return data


# 3. 特征工程
def create_features(df):
    """
    创建时间序列特征（严格防止数据泄露）
    参数:
        df: 原始数据DataFrame
    返回:
        DataFrame: 包含新特征的数据
    """
    print("\n正在创建特征...")

    # 基础时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # 滞后特征（仅使用历史数据）
    for lag in [1, 3, 6, 24]:
        df[f'load_lag_{lag}'] = df['load'].shift(lag)
        df[f'temp_lag_{lag}'] = df['temp'].shift(lag)

    # 滚动特征（使用shift确保无未来数据）
    df['load_rolling_mean_3'] = df['load'].shift(1).rolling(3).mean()
    df['temp_rolling_mean_3'] = df['temp'].shift(1).rolling(3).mean()

    # 删除包含NaN的行
    df = df.dropna()
    print(f"特征创建完成，最终特征数: {len(df.columns) - 2}")  # 减去timestamp和load列
    return df


# 4. 数据准备
def prepare_data(data):
    """准备训练和测试数据"""
    print("\n准备训练/测试数据...")

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(['timestamp', 'load'], axis=1))
    y = data['load'].values

    # 按时间划分训练测试集
    split_idx = int(0.7 * len(X))  # 70%训练，30%测试
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# 5. 模型定义
def initialize_models():
    """初始化所有模型（应用强正则化）"""
    print("\n初始化模型...")

    models = {
        "决策树": DecisionTreeRegressor(
            max_depth=3,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        ),
        "随机森林": RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=20,
            max_samples=0.5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.5,
            colsample_bytree=0.5,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.5,
            colsample_bytree=0.5,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    }
    return models


# 6. 模型训练与评估
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """训练模型并评估性能"""
    print("\n开始模型训练与评估...")
    results = []

    for name, model in models.items():
        print(f"\n正在训练 {name}...")
        model.fit(X_train, y_train)

        # 评估
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results.append({
            'Model': name,
            'Train_R2': r2_score(y_train, y_pred_train),
            'Test_R2': r2_score(y_test, y_pred_test),
            'Test_MSE': mean_squared_error(y_test, y_pred_test),
            'Test_MAE': mean_absolute_error(y_test, y_pred_test),
            'Test_MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100
        })

        # 绘制预测图
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[:100], label='实际值')
        plt.plot(y_pred_test[:100], label='预测值')
        plt.title(f'{name} 预测效果 (前100个样本)')
        plt.legend()
        plt.show()

    return results


# 7. 结果分析与可视化
def analyze_results(results, models, X_test, y_test):
    """分析并可视化结果"""
    print("\n分析结果...")

    # 显示性能指标
    results_df = pd.DataFrame(results)
    print("\n优化后的模型性能:")
    print(results_df.to_string(index=False))

    # 残差分析
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        residuals = y_test - model.predict(X_test)
        sns.kdeplot(residuals, label=name)
    plt.title('残差分布对比')
    plt.legend()
    plt.show()


# 主函数
def main():
    """主执行流程"""
    setup_environment()

    # 数据准备
    data = load_and_preprocess_data('energy.csv')
    data = create_features(data)
    X_train, X_test, y_train, y_test = prepare_data(data)

    # 模型训练与评估
    models = initialize_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # 结果分析
    analyze_results(results, models, X_test, y_test)

    print("\n分析完成！")


if __name__ == "__main__":
    main()