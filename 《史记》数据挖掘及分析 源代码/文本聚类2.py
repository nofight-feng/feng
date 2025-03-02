import datetime
import collections
import jieba
import re
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Tuple
import platform
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置matplotlib以正确显示中文字符
def setup_chinese_font():
    system = platform.system()
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/SimSun.ttc'
        if not os.path.exists(font_path):
            font_path = 'C:/Windows/Fonts/Microsoft YaHei.ttf'
    elif system == 'Darwin':
        font_path = '/System/Library/Fonts/PingFang.ttc'
    else:
        font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'

    if os.path.exists(font_path):
        plt.rcParams['font.family'] = 'SimSun'
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# 从Excel文件加载并验证数据
def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    logger.info("从 %s 加载数据", file_path)
    news_df = pd.read_excel(file_path, header=0)

    categories = {'十二本纪': 0, '十表': 0, '八书': 0, '三十世家': 0, '七十列传': 0}
    errors = 0

    for idx, row in news_df.iterrows():
        if len(str(row['分词后内容'])) < 10:
            errors += 1
        categories[row['类别']] += 1

    logger.info("数据统计: %s", categories)
    logger.info("总记录数: %d, 错误数: %d", len(news_df), errors)

    return news_df, categories

# 从文档中提取TF-IDF特征
def extract_features(docs: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    vectorizer = TfidfVectorizer(max_df=0.7)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(docs))
    words = vectorizer.get_feature_names_out()
    weights = tfidf.toarray()

    return tfidf, words, weights

# 执行K-means聚类和降维
def perform_clustering(weights: np.ndarray, n_clusters: int) -> Tuple[KMeans, np.ndarray]:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    pca = PCA(n_components=2)  # 降维至2D以便于可视化
    reduced_data = pca.fit_transform(weights)
    kmeans.fit(weights)

    # 计算轮廓系数
    silhouette_avg = metrics.silhouette_score(weights, kmeans.labels_)
    logger.info("轮廓系数: %.3f", silhouette_avg)

    return kmeans, reduced_data

# 分析和可视化聚类内容
def analyze_clusters(cluster_lists: List[List[str]], n_clusters: int):
    plt.figure(figsize=(15, 10))

    # 绘制聚类大小
    sizes = [len(cluster) for cluster in cluster_lists]
    plt.subplot(2, 1, 1)
    plt.bar(range(n_clusters), sizes)
    plt.title('各聚类的文本数量分布')
    plt.xlabel('聚类编号')
    plt.ylabel('文本数量')

    # 为每个聚类创建词云
    for i, cluster_docs in enumerate(cluster_lists):
        words = ' '.join(cluster_docs).split()
        word_freq = collections.Counter(words)
        logger.info("聚类 %d: %d 篇文档, 关键词: %s",
                    i, len(cluster_docs), word_freq.most_common(10))

    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 使用plotly创建交互式聚类可视化
def visualize_clusters_interactive(reduced_data: np.ndarray, labels: np.ndarray,
                                   original_data: pd.DataFrame):
    fig = px.scatter(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        color=labels.astype(str),
        hover_data=[original_data['标题'], original_data['类别']],
        title='文本聚类可视化',
        labels={'color': '聚类', 'x': '第一主成分', 'y': '第二主成分'}
    )

    fig.write_html('cluster_visualization.html')

# 主函数
def main():
    setup_chinese_font()

    try:
        # 1. 加载数据
        news_df, categories = load_data('史记分词后.xlsx')

        # 2. 提取特征
        docs = [str(content) for content in news_df['分词后内容']]
        tfidf, words, weights = extract_features(docs)

        # 3. 执行聚类
        n_clusters = 5
        kmeans, reduced_data = perform_clustering(weights, n_clusters)

        # 4. 更新DataFrame中的聚类标签
        news_df['聚类'] = kmeans.labels_

        # 5. 分析聚类
        cluster_lists = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            cluster_lists[label].append(docs[i])

        analyze_clusters(cluster_lists, n_clusters)

        # 6. 创建交互式可视化
        visualize_clusters_interactive(reduced_data, kmeans.labels_, news_df)

        # 7. 创建并显示交叉表
        cross_tab = pd.crosstab(news_df['类别'], news_df['聚类'],
                                margins=True, margins_name='合计')
        logger.info("\n交叉表:\n%s", cross_tab)

        # 8. 保存结果
        news_df.to_excel('史记聚类分析结果.xlsx', index=False)
        cross_tab.to_excel('史记聚类交叉分析.xlsx')

        logger.info("分析成功完成")

    except Exception as e:
        logger.error("发生错误: %s", str(e))
        raise

if __name__ == "__main__":
    main()