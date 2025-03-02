import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 定义一个函数来读取文件并进行分词
def read_and_segment(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    seg_list = jieba.cut(content)
    return seg_list

# 定义一个函数来计算词频
def calculate_frequency(words):
    return Counter(words)

# 定义一个函数来将词频结果写入TXT文件
def write_frequency_to_file(frequencies, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for word, freq in frequencies.items():
            file.write(f"{word}: {freq}\n")

# 调用函数
txt_file_path = '史记分词后.txt'  # 这里填写你的TXT文件路径
segmented_text = read_and_segment(txt_file_path)
frequencies = calculate_frequency(segmented_text)

write_frequency_to_file(frequencies, '史记词频分析.txt')  # 将词频结果保存到word_frequencies.txt

print("词频统计完成，结果已保存到 '史记词频分析.txt'")
# 绘制词云
def plot_wordcloud(word_counts, max_words=200, font_path='C:\\Windows\\Fonts\\simhei.ttf'):
    wordcloud = WordCloud(width=800, height=400, max_words=max_words, font_path=font_path).generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
plot_wordcloud(frequencies)
#
# def plot_wordcloud(word_freq):
#     wordcloud = (
#         WordCloud()
#         .add("", list(word_freq.items()), word_size_range=[20, 100])
#         .set_global_opts(title_opts=opts.TitleOpts(title="《红楼梦》词云"))
#     )
#     wordcloud.render("词云.html")