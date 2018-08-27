import requests
import re
import json
from zhon import hanzi
import csv
import time
import pandas as pd
from tqdm import tqdm

def get_article_info(per_page,page_num):
    """
    获取文章基本信息
    :param per_page:
    :param page_num:
    :return:
    """
    with open('36kr.csv', 'w', encoding='utf-8', newline='') as out_data:
        csv_writer = csv.writer(out_data)
        csv_writer.writerow(('id', 'summary', 'column_name', 'title', 'cover',
                             'published_at', 'extraction_tags', 'favourite_num'))

        for i in range(1,page_num+1):
            # 获取文章基本信息
            url='https://36kr.com/api/search-column/mainsite?per_page={}&page={}'.format(per_page,i)
            res=requests.get(url)
            items=res.json()['data']['items']
            print(len(items),items)
            for item in items:
                if 'favourite_num' not in items:
                    item['favourite_num'] = 0
                csv_writer.writerow((item['id'], item['summary'].replace('\n', ''),
                                     item['column_name'], item['title'],
                                     item['cover'], item['published_at'],
                                     item['extraction_tags'], item['favourite_num']))
            print("已经爬取了{}条文章".format(i * per_page))
            time.sleep(1)

# get_article_info(300,6)


def get_article():
    """
    获取文章内容
    :return:
    """
    article_infos=pd.read_csv('36kr.csv')
    ids=article_infos['id']
    titles=article_infos['title']
    summaries=article_infos['summary']
    contents=[]
    for id in ids.tolist():
        article_url='http://36kr.com/p/{}.html'.format(id)
        print("正在爬取文章{}".format(article_url))
        res=requests.get(article_url)
        try:
            content_pattern=re.compile(r'<script>var props=(.*?),locationnal={')
            data=re.findall(content_pattern,res.text)[0]
            # print(data,type(data))
            data=json.loads(data)
            # print(data['detailArticle|post'])
            # title=data['detailArticle|post']['title']
            # summary=data['detailArticle|post']['summary']

            content=data['detailArticle|post']['content']
            zh_pattern=re.compile(r'[^\u4e00-\u9fa5'+hanzi.punctuation+']') # 中文的编码范围是：\u4e00-\u9fa5 这里保留了中文的标点符号
            content=re.sub(zh_pattern,'',content)
        except Exception as e:
            print(e)
            content=''
        # print(title)
        # print(summary)
        # print(content)
        contents.append(content)
        time.sleep(0.1)

    data=pd.DataFrame()
    data['id']=ids
    data['title']=titles
    data['summary']=summaries
    data['content']=contents
    data.to_csv('36kr_articles.csv',index=False)

get_article()