import jieba
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans,MiniBatchKMeans

def load_articles():
    """
    将文本分词
    :return:
    """
    filename='data/36kr_articles.pkl'
    if os.path.exists(filename):
        with open(filename,'rb') as in_data:
            articles=pickle.load(in_data)
            return articles

    stop_words = [word.strip() for word in open('data/stop_words.txt', 'r',encoding='utf-8').readlines()]
    print(stop_words)
    data = pd.read_csv('data/36kr_articles.csv')

    print("正在分词...")
    # 分词，去除停用词
    data['title'] = data['title'].apply(lambda x: " ".join([word for word in jieba.cut(x) if word not in stop_words]))
    data['summary'] = data['summary'].apply(lambda x: " ".join([word for word in jieba.cut(x) if word not in stop_words]))
    data['content'] = data['content'].apply(lambda x: " ".join([word for word in jieba.cut(str(x)) if word not in stop_words]))

    articles=[]
    for title,summary,content in zip(data['title'].tolist(),
                                     data['summary'].tolist(),
                                     data['content'].tolist()):
        article=title+summary+content
        articles.append(article)

    with open(filename,'wb') as out_data:
        pickle.dump(articles,out_data,pickle.HIGHEST_PROTOCOL)

    return articles


# articles=load_articles()

def transform(articles,n_features=1000):
    """
    提取tf-idf特征
    :param articles:
    :param n_features:
    :return:
    """
    vectorizer=TfidfVectorizer(max_df=0.5,max_features=n_features,min_df=2,use_idf=True)
    X=vectorizer.fit_transform(articles)
    return X,vectorizer

def train(X,vectorizer,true_k=10,mini_batch=False,show_label=False):
    """
    训练 k-means
    :param X:
    :param vectorizer:
    :param true_k:
    :param mini_batch:
    :param show_label:
    :return:
    """
    if mini_batch:
        k_means=MiniBatchKMeans(n_clusters=true_k,init='k-means++',n_init=1,
                                init_size=1000,batch_size=1000,verbose=False)
    else:
        k_means=KMeans(n_clusters=true_k,init='k-means++',max_iter=300,n_init=1,
                       verbose=False)
    k_means.fit(X)
    if show_label: # 显示标签
        print("Top terms per cluster:")
        order_centroids=k_means.cluster_centers_.argsort()[:,::-1]
        terms=vectorizer.get_feature_names()
        # print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d" % i ,end='')
            for ind in order_centroids[i,:10]:
                print(' %s' % terms[ind],end='')
            print()
    result=list(k_means.predict(X))
    print('Cluster distribution:')
    print(dict([(i,result.count(i)) for i in result]))
    return -k_means.score(X)

def test():
    """
    测试选择最优参数
    :return:
    """
    articles=load_articles()
    print("%d docments" % len(articles))
    X,vectorizer=transform(articles,n_features=500)
    true_ks=[]
    scores=[]
    for i in range(3,80,1):
        score=train(X,vectorizer,true_k=i)/len(articles)
        print(i,score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8,4))
    plt.plot(true_ks,scores,label="error",color="red",linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("error")
    plt.legend()
    plt.show()

# test()
def out():
    """
    在最优参数下输出聚类效果
    :return:
    """


    articles = load_articles()
    X,vectorizer = transform(articles,n_features=500)
    score = train(X,vectorizer,true_k=10,show_label=True)/len(articles)
    print (score)

out()