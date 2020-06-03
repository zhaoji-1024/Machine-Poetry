import requests
import re

#定义请求头为全局变量
HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
}

def get_detail_url():
    """
    获取详情页url
    :return: 详情页url以列表形式

    """
    #详情页url前缀
    base_url = "https://www.gushiwen.org/"

    #首页url
    url = "https://www.gushiwen.org/gushi/quantang.aspx"

    #获取首页html代码数据
    html = requests.get(url, HEADERS).text

    #使用正则表达式获取url列表
    detail_urls = re.findall('<span><a href="(.*?)" target', html)

    #使用map映射函数将base_url加到详情页url上
    detail_urls = list(map(lambda x:base_url + x, detail_urls))

    #只获取到第772卷诗
    detail_urls = detail_urls[:-28]

    return detail_urls

def parse_detail_page(url):

    #获取详情页的html数据
    html = requests.get(url, HEADERS).content.decode("utf-8")

    #利用正则表达式提取到诗文数据
    data = re.findall('<div class="contson">(.*?)</div>', html, re.S)[0]

    #利用re.sub()清洗数据
    data = re.sub(r"<.*?>", "", data)

    #获取唐诗标题
    poems_title = re.findall("「(.*?)」", data)[:-1]

    #提前编译正则表达式
    r = re.compile(r"""
                    」 #标题后的特殊字符
                    [\u4e00-\u9fa5]+    #作者姓名
                    \s+    #姓名后的1或多个空格
                    (.*?)    #诗句内容
                    \s+   #末句后得1或多个空格r
                    [\u4e00-\u9fa5]+    #卷这个汉字r
                    \d+  #卷这个字后面得数字
                    \w+?  #下划线
                    """, re.VERBOSE)

    #获取诗句内容
    poems_content = re.findall(r, data)

    #去掉诗句中的空格字符
    poems_content = list(map(lambda x:x.replace(" ", ""), poems_content))

    #将标题和诗句汇总
    poems = list(map(lambda x,y: x + ":" + y, poems_title, poems_content))

    #返回唐诗列表
    return poems


def main():

    #获取详情页url
    urls = get_detail_url()

    #遍历urls获取唐诗数据
    for index, url in enumerate(urls):

        poem = parse_detail_page(url)

        # 写入文件
        with open("./data/poems.txt", "a", encoding="utf-8") as fp:
            for poe in poem:
                fp.write(poe + "\n")

        fp.close()

        print("第{}卷写入成功...".format(index))

if __name__ == '__main__':

    main()