## 基于doc2vec的文本聚类

### 1 训练doc2vec 模型

`doc2vec-sim.py`:该代码主要解释了如何使用gensim训练doc2vec，然后计算文档相似度，
并给出样例：计算出一个文本最相似的10篇文章。

### 2 k-means 实现聚类
加载doc2vec的文档向量作为训练样本，然后给kmeans训练
### 3 数据集

`36kr的文章`

### 4 参考资料：
- [gensim doc2vec + sklearn kmeans 做文本聚类](https://blog.csdn.net/juanjuan1314/article/details/75124046)
- [基于doc2vec的文本聚类](https://blog.csdn.net/weixin_39837402/article/details/80336457)