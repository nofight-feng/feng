import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import platform
import os
import warnings
import logging
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 设置中文字体，以便matplotlib可以正确显示中文字符
def setup_chinese_font():
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = ['SimHei']
    else:
        plt.rcParams['font.family'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# 加载并验证数据集
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        logger.info(f"成功加载 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"文件加载错误: {e}")
        return pd.DataFrame()

# 将文本转换为TF-IDF特征
def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['分词后内容'].astype(str))
    y = df['类别']
    return X, y, vectorizer

# 训练并评估单个模型
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算指标
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    results = {
        'model_name': model_name,
        'accuracy': model.score(X_test, y_test),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    logger.info(f"\n{model_name} 结果：")
    logger.info(f"准确率: {results['accuracy']:.4f}")
    logger.info(f"交叉验证: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
    return results

# 创建分类结果的交互式可视化
def plot_results(results_list: List[Dict], save_path: str = 'classification_results.html'):
    # 创建比较图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('模型准确率比较', '交叉验证分数',
                        '特征重要性', '混淆矩阵')
    )

    # 模型比较
    models = [r['model_name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    cv_means = [r['cv_mean'] for r in results_list]
    cv_stds = [r['cv_std'] for r in results_list]

    # 准确率条形图
    fig.add_trace(
        go.Bar(name='测试准确率', x=models, y=accuracies),
        row=1, col=1
    )

    # 交叉验证图
    fig.add_trace(
        go.Bar(name='CV分数', x=models, y=cv_means,
               error_y=dict(type='data', array=cv_stds * 2)),
        row=1, col=2
    )

    # 更新布局
    fig.update_layout(
        title_text="文本分类结果",
        height=800,
        showlegend=True
    )

    # 保存交互式图表
    fig.write_html(save_path)
    logger.info(f"交互式可视化保存至 {save_path}")

# 主函数
def main():
    # 设置
    setup_chinese_font()

    try:
        # 加载数据
        df = load_data('史记分词后.xlsx')
        if df.empty:
            return

        # 预处理数据
        X, y, vectorizer = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 定义要评估的模型
        models = {
            '逻辑回归': LogisticRegression(max_iter=1000),
            '朴素贝叶斯': MultinomialNB(),
            '线性SVM': LinearSVC(max_iter=2000),
            '随机森林': RandomForestClassifier(n_estimators=100)
        }

        # 训练并评估模型
        results_list = []
        for name, model in models.items():
            try:
                results = train_evaluate_model(
                    model, X_train, X_test, y_train, y_test, name
                )
                results_list.append(results)
            except Exception as e:
                logger.error(f"训练 {name} 时出错: {e}")

        # 创建可视化
        plot_results(results_list)

        # 保存详细结果
        output_df = pd.DataFrame([{
            '模型': r['model_name'],
            '准确率': r['accuracy'],
            'CV分数': r['cv_mean'],
            'CV标准差': r['cv_std'],
            '分类报告': r['classification_report']
        } for r in results_list])

        output_df.to_excel('classification_results.xlsx', index=False)
        logger.info("结果保存至 classification_results.xlsx")

    except Exception as e:
        logger.error(f"发生错误: {e}")

if __name__ == "__main__":
    main()