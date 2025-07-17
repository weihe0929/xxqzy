import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

#数据加载和预处理
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['timestamp'])

    #特征工程
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    #滞后特征
    df['temp_lag1'] = df['temp'].shift(1)
    df['load_lag1'] = df['load'].shift(1)
    df['load_lag24'] = df['load'].shift(24)

    #滑动窗口特征
    df['temp_rolling_24_mean'] = df['temp'].rolling(24).mean()
    df['load_rolling_24_mean'] = df['load'].rolling(24).mean()

    #删除缺失值
    df = df.dropna()

    #标准化
    scaler = StandardScaler()
    numeric_cols = ['temp', 'temp_lag1', 'temp_rolling_24_mean']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

#加载数据
df = pd.read_csv('energy.csv', parse_dates=['timestamp'])

#基本统计
print(df.describe())

#时间序列可视化
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['load'], label='Energy Load')
plt.plot(df['timestamp'], df['temp'], label='Temperature', color='orange')
plt.legend()
plt.title('Energy Load and Temperature Over Time')
plt.show()

#相关性分析
corr = df[['load', 'temp']].corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()
