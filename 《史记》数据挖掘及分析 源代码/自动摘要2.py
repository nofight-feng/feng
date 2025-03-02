import pandas as pd
import numpy as np
from collections import defaultdict
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import re
import warnings
from typing import List, Dict
import os

warnings.filterwarnings('ignore')

# 加载Excel文件并执行基本的数据验证
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"文件加载错误: {e}")
        return pd.DataFrame()

# 清洗和预处理文本
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 移除URL和特殊字符
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# 使用TextRank算法生成摘要
def textrank_summarize(text: str, num_sentences: int = 3) -> str:
    if not text or len(text) < 100:  # 跳过非常短的文本
        return text

    # 将文本分割成句子（假设分词后的句子之间有空格）
    sentences = text.split()
    if len(sentences) <= num_sentences:
        return text

    # 创建相似度矩阵
    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform([' '.join(jieba.cut(sent)) for sent in sentences])
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    except:
        return ' '.join(sentences[:num_sentences])  # 如果失败则返回前n个句子

    # 创建图并计算分数
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # 获取排名最高的句子
    ranked_sentences = [(scores[i], s) for i, s in enumerate(sentences)]
    ranked_sentences.sort(reverse=True)

    return ' '.join([sent for score, sent in ranked_sentences[:num_sentences]])

# 从文本中提取关键词
def extract_keywords(text: str, topK: int = 5) -> List[str]:
    return jieba.analyse.extract_tags(text, topK=topK)

# 对数据集进行文本分析和摘要生成
def analyze_text(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for idx, row in df.iterrows():
        try:
            # 处理每篇文本
            content = str(row['分词后内容'])
            clean_content = preprocess_text(content)

            # 生成摘要
            summary = textrank_summarize(clean_content)

            # 提取关键词
            keywords = extract_keywords(clean_content)

            results.append({
                '序号': row['序号'],
                '类别': row['类别'],
                '标题': row['标题'],
                '关键词': '、'.join(keywords),
                '摘要': summary
            })

            # 打印进度
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1} 篇文档")

        except Exception as e:
            print(f"处理文档 {idx + 1} 时出错: {str(e)}")
            continue

    return pd.DataFrame(results)

# 将结果保存到Excel文件，并进行错误处理
def save_results(df: pd.DataFrame, output_file: str) -> bool:
    try:
        # 如果原始文件被锁定，则尝试保存为一个唯一的文件名
        base, ext = os.path.splitext(output_file)
        counter = 1
        current_file = output_file

        while True:
            try:
                df.to_excel(current_file, index=False)
                print(f"结果成功保存至 {current_file}")
                return True
            except PermissionError:
                current_file = f"{base}_{counter}{ext}"
                counter += 1
                if counter > 10:  # 限制尝试次数
                    raise
            except Exception as e:
                print(f"文件保存错误: {str(e)}")
                return False
    except Exception as e:
        print(f"save_results 函数中发生错误: {str(e)}")
        return False

# 主函数
def main():
    try:
        # 加载数据
        print("正在加载数据...")
        df = load_data('史记分词后.xlsx')
        if df.empty:
            return

        # 执行分析
        print("\n正在生成摘要和提取关键词...")
        results_df = analyze_text(df)

        # 保存结果
        output_file = '史记文本摘要.xlsx'
        if save_results(results_df, output_file):
            # 显示样本结果
            print("\n样本结果（前3篇文档）：")
            for _, row in results_df.head(3).iterrows():
                print(f"\n标题: {row['标题']}")
                print(f"类别: {row['类别']}")
                print(f"关键词: {row['关键词']}")
                print(f"摘要: {row['摘要']}")
                print("-" * 80)

    except Exception as e:
        print(f"main 函数中发生错误: {str(e)}")

if __name__ == "__main__":
    main()