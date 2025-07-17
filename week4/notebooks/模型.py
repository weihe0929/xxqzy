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
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 1. 数据加载与预处理
data = pd.read_csv('energy.csv', parse_dates=['timestamp'])
data = data.sort_values('timestamp').reset_index(drop=True)


# 2. 严格的特征工程（防止数据泄露）
def create_features(df):
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

    return df.dropna()


data = create_features(data)

# 3. 数据标准化（改用StandardScaler）
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(['timestamp', 'load'], axis=1))
y = data['load'].values

# 4. 时间序列分割（严格按时间划分）
split_idx = int(0.7 * len(X))  # 70%训练，30%测试
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 5. 模型定义与训练（更强的正则化）
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

# 6. 训练与评估
results = []
for name, model in models.items():
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
    plt.plot(y_test[:100], label='Actual')
    plt.plot(y_pred_test[:100], label='Predicted')
    plt.title(f'{name} Prediction (First 100 Samples)')
    plt.legend()
    plt.show()

# 7. 结果分析
results_df = pd.DataFrame(results)
print("\n优化后的模型性能:")
print(results_df.to_string(index=False))

# 8. 残差分析
plt.figure(figsize=(10, 6))
for name, model in models.items():
    residuals = y_test - model.predict(X_test)
    sns.kdeplot(residuals, label=name)
plt.title('Residual Distribution Comparison')
plt.legend()
plt.show()

# 9. 检查数据泄露（关键步骤）
corr_matrix = pd.DataFrame(X_train).corrwith(pd.Series(y_train))
print("\n特征与目标变量的相关系数:")
print(corr_matrix.sort_values(ascending=False).head(10))