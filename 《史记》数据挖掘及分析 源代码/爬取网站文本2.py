# coding  : utf-8
# fun     : 爬取史记所有的章节标题和章节内容
# @Author : Labyrinthine Leo
# @Time   : 2021.02.01

import requests
from bs4 import BeautifulSoup
from time import sleep
from openpyxl import Workbook

def book_spider(url):
    """
    爬取史记文本信息
    :param url: 网站url
    :return:
    """
    # 1.指定url
    url = url
    # 2.UA伪装
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
    }
    # 3.发送请求&获取响应数据
    page_text = requests.get(url=url, headers=headers)
    page_text.encoding = page_text.apparent_encoding # 获取编码
    page_text = page_text.text
    # 4.对text页面进行章节标题文本提取并获取每个章节对应的url链接
    soup = BeautifulSoup(page_text, 'lxml')
    aTagList = soup.select('a.tabli') # 获取a标签信息
    titleList = [i.text for i in aTagList] # 获取a标签中的文本信息
    urlList = [i["href"] for i in aTagList] # 获取s标签中每个章节的url

    # 创建Excel工作簿
    wb = Workbook()
    ws = wb.active
    ws.title = "史记章节"

    # 写入表头
    ws.append(["序号", "标题", "网址", "内容"])

    # 5.保存章节内容
    for index, chp in enumerate(zip(titleList, urlList), start=1):
        write_chapter(index, chp, ws)
        sleep(5)

    # 保存Excel文件
    wb.save("史记章节.xlsx")
    print("已成功下载史记全文并保存到Excel文件！")

def write_chapter(index, content_list, ws):
    """
    将每章节信息提取并写入Excel
    :param index: 序号
    :param content_list: 包含标题和链接的元组
    :param ws: Excel工作表对象
    :return:
    """
    # 获取标题和链接
    title, url = content_list
    intact_url = "https://www.shicimingju.com" + url
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
    }
    page_text = requests.get(url=intact_url, headers=headers, timeout=10)
    page_text.encoding = page_text.apparent_encoding  # 获取编码
    page_text = page_text.text
    soup = BeautifulSoup(page_text,'lxml')
    content = soup.select('.text.p_pad')
    txt = "" # 构建文本字符串
    for i in content:
        txt += i.text

    # 写入Excel
    ws.append([index, title, intact_url, txt])
    print("已成功下载{}内容".format(title.split('·')[0]))

if __name__ == '__main__':
    url = "https://www.shicimingju.com/book/shiji.html"
    book_spider(url)