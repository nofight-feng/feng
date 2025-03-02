import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import jieba
import warnings

warnings.filterwarnings('ignore')

# 读取Excel文件
df = pd.read_excel('史记分词后.xlsx')

# 定义主题数量
n_topics = 5

# 创建结果列表
results = []

# 对所有文本进行预处理
texts = df['分词后内容'].fillna('').tolist()

try:
    # 创建更适合单文档分析的TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,  # 减少特征数量
        max_df=1.0,  # 允许所有词频
        min_df=1,  # 允许仅出现一次的词
        token_pattern=r'(?u)\b\w+\b'  # 允许单字词
    )

    # 首先对整个语料库进行拟合
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # 创建LDA模型
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10
    )

    # 对整个语料库进行LDA分析
    lda_output = lda.fit_transform(tfidf_matrix)

    # 获取特征名称
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 对每个章节进行处理
    for idx, row in df.iterrows():
        try:
            # 获取当前文档的LDA结果
            doc_topics = lda_output[idx]

            # 获取主导主题
            dominant_topic = np.argmax(doc_topics)
            topic_probability = doc_topics[dominant_topic]

            # 获取该主题的关键词
            topic = lda.components_[dominant_topic]
            top_keywords_idx = topic.argsort()[:-5 - 1:-1]
            top_keywords = [feature_names[i] for i in top_keywords_idx]

            # 存储结果
            results.append({
                '序号': row['序号'],
                '类别': row['类别'],
                '标题': row['标题'],
                '主导主题': f'主题{dominant_topic + 1}',
                '主题概率': f'{topic_probability:.3f}',
                '主题关键词': ', '.join(top_keywords)
            })

        except Exception as e:
            print(f"处理章节 {row['标题']} 时出错: {str(e)}")
            # 添加空结果以保持一致性
            results.append({
                '序号': row['序号'],
                '类别': row['类别'],
                '标题': row['标题'],
                '主导主题': '处理失败',
                '主题概率': '0.000',
                '主题关键词': '无法提取关键词'
            })

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    # 保存到Excel文件
    output_file = 'shiji_topics.xlsx'
    results_df.to_excel(output_file, index=False)

    print(f"分析完成，结果已保存至 {output_file}")
    print("\n前5条分析结果：")
    print(results_df.head().to_string())

except Exception as e:
    print(f"发生错误: {str(e)}")
    print("请检查输入数据格式是否正确，并确保'分词后内容'列存在且包含有效文本。")