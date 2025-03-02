import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import jieba
import warnings

warnings.filterwarnings('ignore')

# 读取Excel文件
df = pd.read_excel('史记分词后.xlsx')

# 确保'date'列存在并且是datetime格式
if 'date' not in df.columns:
    df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
else:
    df['date'] = pd.to_datetime(df['date'])

# 创建TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer(max_features=1000, token_pattern=r'(?u)\b\w+\b')

# 预测趋势的函数
def predict_trend(series, steps=7):
    model = ARIMA(series, order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=steps)
    return forecast

# 处理每个章节
results = []
for idx, row in df.iterrows():
    text = row['分词后内容']

    # TF-IDF转换
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 获取前10个关键词
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_indices = tfidf_scores.argsort()[-10:][::-1]
    top_keywords = [feature_names[i] for i in top_indices]

    # 模拟历史数据（如果有真实数据，替换此部分）
    historical_data = np.random.rand(30)  # 模拟30天的数据
    scaler = MinMaxScaler()
    historical_data_scaled = scaler.fit_transform(historical_data.reshape(-1, 1)).flatten()

    # 预测趋势
    trend_prediction = predict_trend(historical_data_scaled)

    # 判断是否为热点
    is_hot = trend_prediction[-1] > historical_data_scaled.mean()

    results.append({
        '序号': row['序号'],
        '类别': row['类别'],
        '标题': row['标题'],
        '预测热度': f'{trend_prediction[-1]:.3f}',
        '是否热点': '是' if is_hot else '否',
        '关键词': ', '.join(top_keywords)
    })

# 创建结果DataFrame
results_df = pd.DataFrame(results)

# 保存至Excel文件
output_file = '史记热点预测.xlsx'
results_df.to_excel(output_file, index=False)

print(f"分析完成，结果已保存至 {output_file}")
print("\n前5条分析结果：")
print(results_df.head().to_string())