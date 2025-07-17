import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 读取数据
data = pd.read_csv('nigerian-songs.csv')

# 数据预处理
# 检查缺失值
print("缺失值统计:\n", data.isnull().sum())

# 处理缺失值 - 这里简单用众数填充分类变量，中位数填充数值变量
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# 编码分类变量
label_encoders = {}
for column in ['artist_top_genre']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 特征工程
# 将release_date转换为年份
data['release_year'] = pd.to_datetime(data['release_date'], errors='coerce').dt.year
data['release_year'].fillna(data['release_year'].median(), inplace=True)

# 将length从毫秒转换为分钟
data['length_min'] = data['length'] / 60000

# 删除不需要的列
data.drop(['name', 'album', 'artist', 'release_date', 'length'], axis=1, inplace=True)

# 定义特征和目标变量
X = data.drop('artist_top_genre', axis=1)
y = data['artist_top_genre']

# 第一组特征选择 - 使用SelectKBest选择前5个特征
selector1 = SelectKBest(f_classif, k=5)
X_selected1 = selector1.fit_transform(X, y)
selected_features1 = X.columns[selector1.get_support()].tolist()
print("\n第一组选择的特征:", selected_features1)

# 第二组特征选择 - 基于相关性选择
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
X_selected2 = X.drop(to_drop, axis=1)
selected_features2 = X_selected2.columns.tolist()
print("第二组选择的特征:", selected_features2)

# 划分训练集和测试集
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_selected1, y, test_size=0.2, random_state=42)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_selected2, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'LGBM': LGBMClassifier(random_state=42)
}


# 修改后的交叉验证和模型评估函数
def evaluate_models(X_train, X_test, y_train, y_test, feature_set_name, feature_names):
    results = {}
    for name, model in models.items():
        # 交叉验证
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # 存储结果
        results[name] = {
            'CV Mean Accuracy': np.mean(cv_scores),
            'CV Std': np.std(cv_scores),
            'Test Accuracy': accuracy,
            'Classification Report': report,
            'Model': model,  # 保存训练好的模型
            'Feature Names': feature_names  # 保存特征名称
        }

        # 如果是决策树，可视化
        if name == 'Decision Tree':
            plt.figure(figsize=(20, 10))
            plot_tree(model, filled=True, feature_names=feature_names,
                      class_names=label_encoders['artist_top_genre'].classes_,
                      rounded=True, proportion=True)
            plt.title(f'Decision Tree Visualization - {feature_set_name}')
            plt.show()

    return results


# 评估第一组特征
print("\n评估第一组特征...")
results1 = evaluate_models(X_train1, X_test1, y_train1, y_test1, 'Feature Set 1', selected_features1)

# 评估第二组特征
print("\n评估第二组特征...")
results2 = evaluate_models(X_train2, X_test2, y_train2, y_test2, 'Feature Set 2', selected_features2)


# 显示结果
def display_results(results, feature_set_name):
    print(f"\n{feature_set_name} 结果:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"交叉验证平均准确率: {metrics['CV Mean Accuracy']:.4f} (±{metrics['CV Std']:.4f})")
        print(f"测试集准确率: {metrics['Test Accuracy']:.4f}")
        print("\n分类报告:")
        print(metrics['Classification Report'])


display_results(results1, "第一组特征")
display_results(results2, "第二组特征")


# 修改后的特征重要性分析函数
def plot_feature_importance(model_info, title):
    model = model_info['Model']
    feature_names = model_info['Feature Names']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"{title} 模型没有特征重要性属性")
        return

    # 确保特征名称和重要性值长度一致
    if len(importances) != len(feature_names):
        print(f"警告: {title} 的特征数量({len(importances)})与特征名称数量({len(feature_names)})不匹配")
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


# 特征重要性分析
for name in models.keys():
    plot_feature_importance(results1[name], f'{name} - 特征重要性 (第一组特征)')
    plot_feature_importance(results2[name], f'{name} - 特征重要性 (第二组特征)')


# 模型对比可视化
def plot_model_comparison(results1, results2):
    models = list(results1.keys())
    acc1 = [results1[m]['Test Accuracy'] for m in models]
    acc2 = [results2[m]['Test Accuracy'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, acc1, width, label='第一组特征')
    rects2 = ax.bar(x + width / 2, acc2, width, label='第二组特征')

    ax.set_ylabel('准确率')
    ax.set_title('不同特征组下模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    fig.tight_layout()
    plt.show()


plot_model_comparison(results1, results2)