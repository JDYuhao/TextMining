# TextMining
# textClassification
本程序是利用python开发的文本挖掘系统。包括分类，聚类，主题模型，深度神经网络。

textClassification.py文本分类器
该文件主要包含现文本特征选择，特征赋权，文本分类三个主要模块
1. 文本特征选择：
    目前文本特征选择是基于词袋(bag-of-words)实现的，对于中文文本，
我采用<a href="https://github.com/fxsjy/jieba/tree/jieba3k">jieba</a>分词完成分词。
因此，使用前请熟悉jieba的使用方法。
    特征选择方面，目前实现了基于卡方，基于信息增益，WLLR, WFO等方法, 我们通过排序的
方案最后获得最重要的前k个词。

2. 文本特征赋权：
    目前实现了one-hot, local tf-idf, global tf-idf 方法转化为特征向量,
    这里是利用<a href = "http://www.numpy.org">numpy</a>包来实现的.

3. 文本分类:
   分类方法主要基于<a href = "http://scikit-learn.org">sklearn</a>包实现的，
   所以使用前请熟悉sklearn包,
   目前实现了naive bayes, multinomial naive bayes, svm, SGD等方法

已经考虑过使用主题模型进行过分类处理，结果分类效果并没有达到词袋模型的结果。
目前正在考虑使用深度神经网络的方法进行文本分类处理，正在考虑RNN, CNN和word2vec。
    
  
  


