import pandas as pd
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from typing import List, Dict
import seaborn as sns

# 从Excel文件加载数据
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path, header=0)
        print(f"成功从 {file_path} 加载 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"文件加载错误: {e}")
        return pd.DataFrame()

# 检查数据集中的错误和类别分布
def check_data(df: pd.DataFrame) -> Dict[str, int]:
    categories = {'八书': 0, '十表': 0, '十二本纪': 0, '三十世家': 0, '七十列传': 0}
    errors = 0
    for _, row in df.iterrows():
        if len(str(row['分词后内容'])) < 10:
            errors += 1
        categories[row['类别']] += 1
    print(f"类别分布: {categories}")
    print(f"总记录数: {len(df)}, 错误数: {errors}")
    return categories

# 计算文本的情感倾向得分
def calculate_sentiment(text: str) -> float:
    words = text.split()
    return sum(SnowNLP(word).sentiments for word in words) / len(words)

# 根据得分将情感倾向进行分类
def categorize_sentiment(score: float, method: str, sentiment_min: float = 0, sentiment_max: float = 1) -> str:
    if method == 'fixed':
        boundaries = [0.2, 0.4, 0.6, 0.8]
    elif method == 'dynamic':
        range_size = (sentiment_max - sentiment_min) / 5
        boundaries = [sentiment_min + i * range_size for i in range(1, 5)]
    else:
        raise ValueError("方法无效。请使用 'fixed' 或 'dynamic'.")

    categories = ['1.极端负面', '2.偏向负面', '3.情感中性', '4.偏向正面', '5.极端正面']
    for i, boundary in enumerate(boundaries):
        if score < boundary:
            return categories[i]
    return categories[-1]

# 对数据集中的文本进行情感分析
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df['情感正向概率'] = df['分词后内容'].apply(calculate_sentiment)
    sentiment_min, sentiment_max = df['情感正向概率'].min(), df['情感正向概率'].max()

    df['情感归类1'] = df['情感正向概率'].apply(lambda x: categorize_sentiment(x, 'fixed'))
    df['情感归类2'] = df['情感正向概率'].apply(lambda x: categorize_sentiment(x, 'dynamic', sentiment_min, sentiment_max))

    print(f"情感得分范围: {sentiment_min:.4f} 至 {sentiment_max:.4f}")
    return df

# 创建交叉表以分析不同类别的情感分布
def create_crosstab(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab(df[column], df['类别'], margins=True, margins_name='合计')

# 绘制情感分布图
def plot_sentiment_distribution(df: pd.DataFrame, column: str, title: str):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, hue='类别')
    plt.title(title)
    plt.xlabel('情感类别')
    plt.ylabel('计数')
    plt.xticks(rotation=45)
    plt.legend(title='类别', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    plt.show()

# 主函数
def main():
    df = load_data('史记分词后.xlsx')
    if df.empty:
        return

    check_data(df)

    df = analyze_sentiment(df)

    for method in ['情感归类1', '情感归类2']:
        print(f"\n{method} 的交叉表：")
        print(create_crosstab(df, method))

    plot_sentiment_distribution(df, '情感归类1', '情感分布（固定边界）')
    plot_sentiment_distribution(df, '情感归类2', '情感分布（动态边界）')

    output_file = '史记情感分析后.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n结果已保存至 {output_file}")

if __name__ == "__main__":
    main()