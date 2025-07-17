import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv('energy.csv', parse_dates=['timestamp'])
data = data.sort_values('timestamp')

# 特征工程 - 提取时间特征
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['day_of_month'] = data['timestamp'].dt.day
data['month'] = data['timestamp'].dt.month
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# 创建滞后特征
data['temp_lag1'] = data['temp'].shift(1)
data['temp_lag2'] = data['temp'].shift(2)
data['temp_lag3'] = data['temp'].shift(3)
data['load_lag1'] = data['load'].shift(1)
data['load_lag2'] = data['load'].shift(2)
data['load_lag3'] = data['load'].shift(3)

# 创建滚动统计特征
data['temp_rolling_mean_3h'] = data['temp'].rolling(window=3).mean()
data['load_rolling_mean_3h'] = data['load'].rolling(window=3).mean()

# 删除包含NaN的行
data = data.dropna()

# 定义特征和目标变量
X = data.drop(['timestamp', 'load'], axis=1)
y = data['load']

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 定义评估函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R2 Score: {r2:.2f}')

    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title('Actual vs Predicted Load (First 100 samples)')
    plt.legend()
    plt.show()

    return mse, mae, r2


# 1. 决策树模型
print("决策树模型:")
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# 决策树可视化
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, max_depth=2)
plt.title("决策树可视化 (前两层)")
plt.show()

# 评估决策树
dt_metrics = evaluate_model(dt_model, X_test, y_test)

# 时间序列分析
plt.figure(figsize=(15, 8))

# 每小时平均负荷
plt.subplot(2, 2, 1)
data.groupby('hour')['load'].mean().plot()
plt.title('每小时平均负荷')
plt.xlabel('小时')
plt.ylabel('负荷')

# 每周每天平均负荷
plt.subplot(2, 2, 2)
data.groupby('day_of_week')['load'].mean().plot()
plt.title('每周每天平均负荷')
plt.xlabel('星期几 (0=周一)')
plt.ylabel('负荷')

# 每月平均负荷
plt.subplot(2, 2, 3)
data.groupby('month')['load'].mean().plot()
plt.title('每月平均负荷')
plt.xlabel('月份')
plt.ylabel('负荷')

# 温度与负荷的关系
plt.subplot(2, 2, 4)
plt.scatter(data['temp'], data['load'], alpha=0.3)
plt.title('温度与负荷的关系')
plt.xlabel('温度')
plt.ylabel('负荷')

plt.tight_layout()
plt.show()