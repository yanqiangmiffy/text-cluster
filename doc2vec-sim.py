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
from sklearn.cluster import KMeans
import os
import pandas as pd
import pickle
import jieba
import random

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
    计算文本之间的相似度
    测试训练的模型， 输出相似文章
    下面做了简单测试，假如我们测试下articles第一个doc相似的10篇文章，然后我们可以看到原文相似度最高
    :return:
    """
    model_dm = Doc2Vec.load("model/doc2vec.model")
    test_text=articles[random.randint(1,1800)]
    print(test_text)
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    for index, sim in sims:
        sentence = articles[index]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(sim,articles[index])


def cluster(articles):
    """
    doc2vec +  kmeans 做文本聚类
    :return:
    """
    articles = [gensim.models.doc2vec.TaggedDocument(articles[i], [i]) for i in range(len(articles))]
    infered_vectors_list=[]
    print("load doc2vec model...")
    model_dm=Doc2Vec.load('model/doc2vec.model')
    print("load train vectors...")
    for text,label in articles:
        vector=model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
    print("train k-mean model ...")
    kmean_model=KMeans(n_clusters=8)
    kmean_model.fit(infered_vectors_list)
    labels = kmean_model.predict(infered_vectors_list[0:1800])
    cluster_centers = kmean_model.cluster_centers_
    with open('model/classification.txt','w',encoding='utf-8') as out_f:
        for i in range(1800):
            string = ""
            text = articles[i][0]
            for word in text:
                string = string + word
            string = string + '\t'
            string = string + str(labels[i])
            string = string + '\n'
            out_f.write(string)

if __name__ == '__main__':
    articles = load_articles()
    # train_doc2vec(articles)
    most_sim_docs(articles)
    # cluster(articles)