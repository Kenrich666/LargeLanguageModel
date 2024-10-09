import urllib.request  # 导入urllib.request库

url = (  # 定义下载URL
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]  # 获取文件名
urllib.request.urlretrieve(url, filename)  # 下载文件
