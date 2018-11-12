# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: doc2vec-kmens.py 
@Time: 2018/11/12 16:12
@Software: PyCharm 
@Description:
"""
import gensim
import numpy as np
from gensim.models.doc2vec import  Doc2Vec,LabeledSentence
from gensim.models.doc2vec import TaggedDocument
import os
import pandas as pd
import pickle
import jieba


def load_articles():
    """
    将文本分词
    :return:
    """

    def split_artilce(article):
        return [word for word in article.split(' ') if word != '']

    filename='data/36kr_articles.pkl'
    if os.path.exists(filename):
        with open(filename,'rb') as in_data:
            articles=pickle.load(in_data)
            articles = [split_artilce(article) for article in articles]
            return articles

    stop_words = [word.strip() for word in open('data/stop_words.txt', 'r',encoding='utf-8').readlines()]
    print(stop_words)
    data = pd.read_csv('data/36kr_articles.csv')

    print("正在分词...")
    # 分词，去除停用词
    data['title'] = data['title'].apply(lambda x: " ".join([word for word in jieba.cut(x)
                                                            if word not in stop_words and x]))
    data['summary'] = data['summary'].apply(lambda x: " ".join([word for word in jieba.cut(x)
                                                                if word not in stop_words and x]))
    data['content'] = data['content'].apply(lambda x: " ".join([word for word in jieba.cut(str(x))
                                                                if word not in stop_words and x]))

    articles=[]
    for title,summary,content in zip(data['title'].tolist(),
                                     data['summary'].tolist(),
                                     data['content'].tolist()):
        article=title+summary+content
        articles.append(article)

    with open(filename,'wb') as out_data:
        pickle.dump(articles,out_data,pickle.HIGHEST_PROTOCOL)
    articles = [split_artilce(article) for article in articles]
    return articles


def train_doc2vec(articles):
    tag_tokenized=[gensim.models.doc2vec.TaggedDocument(articles[i],[i]) for i in range(len(articles))]

    model =Doc2Vec(size=200, min_count=2, iter=200)
    model.build_vocab(tag_tokenized)
    model.train(tag_tokenized, total_examples=model.corpus_count, epochs=model.iter)

    # 保存模型
    model.save('model/doc2vec.model')


def most_sim_docs(articles):
    """
    测试下训练的模型， 输出相似文章
    下面做了简单测试，假如我们测试下articles第一个doc相似的10篇文章，然后我们可以看到原文相似度最高
    :return:
    """
    model_dm = Doc2Vec.load("model/doc2vec.model")
    test_text = ['越南', '司法部', '提了', '类', '加密', '币', '监管', '方案', '部门', '有法可依这份', '报告', '越南政府', '进一步', '订立', '法规', '基础编者按', '本文', '氪', '战略', '合作', '区块', '链', '媒体', '星球', '日报', '公众', '号', '下载', '越南', '司法部', '提了', '类', '加密', '币', '监管', '方案', '部门', '有法可依', '中国', '信息', '通信', '研究院', '发布', '区块', '链', '白皮书', '数据', '显示', '全球', '共有', '家', '公司', '活跃', '区块', '链', '产业', '生态', '中', '美国', '中国', '英国', '链企', '数量', '分列', '前三位', '新加坡', '越南', '泰国', '东南亚', '国家', '渐渐', '区块', '链新', '势力', '越来越', '项目', '业界', '咖', '纷纷', '越南', '路演', '神', '兄弟', '月份', '越南', '发生', '一笔', '超大规模', '欺诈案', '涉案', '金额', '亿美元', '越南', '总理', '阮春福', '签署', '指令', '政府', '机构', '央行', '金融机构', '虚拟',
                 '货币', '活动', '管理', '禁止', '信贷', '机构', '支付', '公司', '数字', '货币', '非法交易', '阮春福', '司法部', '研究', '制定', '统一', '虚拟', '货币', '虚拟', '财产', '电子货币', '管理', '法律', '框架', '这一', '法律', '框架', '变得', '明朗', '报道', '越南', '司法部', '河内', '越南', '首都', '政府', '提交', '一份', '报告', '报告', '评估', '国外', '加密', '货币', '行业', '监管', '框架', '提出', '加密', '货币', '法规', '可选', '方案', '世界', '各国', '三类', '方案', '第一种', '方案', '称为', '浮动', '意', '实施', '宽松', '监管', '制度', '第二种', '禁止', '第三种', '条件', '加密', '货币', '交易', '合法化', '越南', '民事', '经济法', '司', '司长', '指出', '报告', '方案', '利弊', '做', '分析', '这份', '报告', '越南政府', '进一步', '实施', '基础', '越南', '网', '报道', '称', '补充', '说', '政府', '选择', '支持', '方案', '相关', '部委', '建立', '法律', '框架', '管理', '数字', '资产', '货币', '河内', '平衡', '风险', '加密', '货币', '相关', '潜力', '确保', '投资者', '受益', '支持', '加密技术', '发展', '预知', '这份', '提案', '提出', '区块', '链', '加密', '货币', '相关', '行业', '迎来', '有法可依', '此前', '政府', '机构', '加密', '货币', '表态', '报道', '中', '提到', '越南', '工业', '贸易部', '总理', '办公室', '提交', '一份', '提案', '解除', '矿机', '进口', '禁令', '这一', '禁令', '月份', '实施', '禁令', '相关', '企业', '影响', '应先', '设备', '研究', '分类', '越南', '央行', '态度', '拒绝', '承认', '比特', '币', '相关', '加密', '货币', '支付', '手段', '合法性', '越南', '证券监管', '机构', '告诉', '公司', '投资', '基金', '远离', '加密', '货币', '作者', '黄雪姣', '区块', '链', '项目', '报道', '交流', '可加', '微信', '劳请', '备注', '职务', '事由']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    for index, sim in sims:
        sentence = articles[index]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(sim,articles[index])

if __name__ == '__main__':
    articles = load_articles()
    most_sim_docs(articles)