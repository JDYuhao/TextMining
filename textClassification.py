#-*- coding: utf-8 -*-
import os
import jieba
from jieba import analyse
import numpy as np
import re
import math
import heapq
import codecs
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB



def wordSplit(content, punctuation):
    string = ""
    for word in jieba.cut(re.sub(punctuation, "",content),cut_all=False):
        if len(word) > 1:
            string += " %s"%word
    return string[1:]

def mdfLog(number):
    try:
        result=math.log(number)
    except ValueError, e:
        result = 0.0
    return result


class textClassfier:
    def __init__(self):
        self.classDis = {}#记录类的情况
        self.wordInClassDis = {}#记录词语在各类的分布的情况
        self.wordDis = {}#记录词语的分布情况
        self.wordFreq = {}#记录词频
        self.corpus = []#记录训练用的语料
        self.testCorpus = []#记录用于测试的语料
        self.features = {}#记录利用训练集产生后的文本
        self.featureArray = []#记录训练集的转换后的向量

    def getCorpusFromFile(self, filename):
        '''
        该函数通过读入文本，得到分词结果的语料库
        '''
        #处理掉特殊符号
        punctuation = re.compile("[a-zA-Z0-9\s+\.\!\/_,$%^*(+\"\']+|[-+——！，。？、~@#￥%……&*（）]+".decode('utf-8'))

        #对关键词提高权重
        with open('keywords.txt', 'r') as parse_file: 
            for eachline in parse_file:
                jieba.add_word(eachline.strip())

        #处理停用词
        jieba.analyse.set_stop_words('stop_words.txt')


        #读入训练集数据
        with open(filename, "rb") as fp:
            trainData = [line.strip().split("\t") for line in fp]

 
        #分词处理
        self.corpus = [[wordSplit(content[0], punctuation), int(content[1])]
              for content in trainData]
        

    def getTextStat(self):
        '''
        该函数通过语料库，得到基本相关的统计结果
        '''
        #开始计数过程
        index = 0
        for content in self.corpus:
            #类分布计数
            index += 1
            if content[1] not in self.classDis:
                self.classDis[content[1]] = 0
            self.classDis[content[1]]+=1

            allWords = content[0].split(" ")

            for word in allWords:
                #词频分布计数
                if word not in self.wordFreq:
                    self.wordFreq[word] = 0
                self.wordFreq[word]+=1
            
            for word in set(allWords):
                if word == '':
                    continue
                
                #词语分布计数
                if word not in self.wordDis:
                    self.wordDis[word] = 0
                self.wordDis[word]+=1
                
                #词语在各类的分布计数
                if (word, content[1]) not in self.wordInClassDis:
                    self.wordInClassDis[(word, content[1])] = 0
                self.wordInClassDis[(word, content[1])]+=1


    def featureSelection(self, k=200,  method = 'chi'):
        '''
        这个函数根据各种统计量来选择相应的特征
        '''
        wordWgt = {}
        for word in self.wordDis.keys():

            weight = 0.0
            totalLen = len(self.corpus)
            
            for wordClass in self.classDis.keys():
                if (word, wordClass) not in self.wordInClassDis:
                    A = 0.0
                else:
                    A = float(self.wordInClassDis[(word, wordClass)])
                B = float(self.wordDis[word]-A)
                C = float(self.classDis[wordClass]-A)
                D = float(len(self.corpus)-A-B-C)


                #利用卡方统计做feature selection
                if method == 'chi':
                    weight += (A+C)*(A*D-C*B)*(A*D-C*B)\
                       /((A+C)*(B+D)*(A+B)*(C+D))

                #利用信息增益做feature selection
                if method == 'infoGain':
                    weight -= (A+B)/totalLen*(A/(A+B)*mdfLog(A/(A+B)))+(C+D)/totalLen\
                    *(C/(C+D)*mdfLog(C/(C+D)))

                #利用WLLR做feature selection
                if method == 'WLLR':
                    weight += (A+C)/totalLen*(A/(A+C))*(mdfLog(A*(B+D))-mdfLog(B*(A+C)))

                #利用WFO做feature selection
                if method == 'WFO':
                    lda = 0.15
                    if ((A/(A+B)) > (C/(C+D))):
                        weight += (A+C)/totalLen*\
                                  math.pow(A/(A+B), lda)*math.pow(mdfLog(A/(A+B))-mdfLog(C/(C+D)), 1-lda)
                    else:
                        weight = 0.0
            wordWgt[word] = weight
        for selectWord, weight in heapq.nlargest(k, wordWgt.items(), key=lambda x:x[1]):
                self.features[selectWord] = 0

    def featureToWeightedArray(self, method = 'localTfIdf'):
        '''
        这一步根据每一个文本构建出一个向量，用于训练分类器
        '''
        self.featureArray = []
        for content in self.corpus:
            tmpDict = self.features.copy()
            allWords = content[0].split(" ")

            if method == 'bool':
                for word in allWords:
                    if word in tmpDict:
                        tmpDict[word] = 1

            if method == 'localTfIdf':
                for word in allWords:
                    if word in tmpDict:
                        tmpDict[word]+=1.0
                        
                for word in allWords:
                    if word in tmpDict and tmpDict[word] > 1e-6:
                        tmpDict[word] = tmpDict[word]/len(allWords)\
                                *mdfLog(1.0*sum(self.wordFreq.values())/self.wordFreq[word])

            if method =='globalTfIdf':
                splitWords = jieba.analyse.extract_tags(content[0], topK=100,
                                                        withWeight=True, allowPOS=())
                for word, weight in splitWords:
                    if word in tmpDict:
                        tmpDict[word] = weight
                                    
            self.featureArray.append(tmpDict.values())
                
        self.featureArray = np.array(self.featureArray)

    def trainClassifer(self, classifer = 'svm'):
        '''
        这一步是基本的模型选择器，在测试哪个模型的性能最好
        '''

        #确定训练集的位置
        if classifer != 'MNB':
            self.featureArray = preprocessing.scale(self.featureArray)
       
        allTarget = np.array(map(lambda x:x[1], self.corpus))


        #确定使用的算法
        if classifer == 'svm':
            clf = SVC(kernel='rbf')
        if classifer == 'SGD':
            clf = SGDClassifier()
        if classifer == 'GNB':
            clf = GaussianNB()
        if classifer == 'MNB':
            clf = MultinomialNB()
            
        clf.fit(self.featureArray, allTarget)
        #这里使用了准确率的指标
        return clf, self.featureArray.mean(axis=0), self.featureArray.std(axis=0)

    def prediction(self, testFilename, toArrMtd = 'bool', classifer = 'svm'):
        clf, mean, std = self.trainClassifer(classifer)
        self.getCorpusFromFile(testFilename)
        self.featureToWeightedArray(toArrMtd)
        if classifer!= 'MNB':
            for k in range(self.featureArray.shape[0]):
                self.featureArray[k,:] = (self.featureArray[k,:]-mean)/std
        with open('classResult.txt', 'w') as f:
            for result in clf.predict(self.featureArray):
                f.write(str(result)+'\n')
        
    
if __name__=="__main__":
    #参数设置
    trainFile = "trainText.txt" #训练文本，分为两列，第一列为文本，第二列为类别，用数值
    testFile = "testText.txt" #测试文本，只有文本
    featureSelectionMethod = 'chi' #特征选择方法
    numFeatures = 1000 #选择出的特征数
    toArrMtd = 'localTfIdf'#特征赋权的方法
    myClassifer = 'svm' #确定分类器

    #通过训练文本做特征选择和特征赋权
    nbClassifer = textClassfier()
    nbClassifer.getCorpusFromFile("trainText.txt")
    nbClassifer.getTextStat()
    nbClassifer.featureSelection(numFeatures, method =featureSelectionMethod)
    nbClassifer.featureToWeightedArray(method = toArrMtd)
    nbClassifer.trainClassifer(classifer = myClassifer)
    nbClassifer.prediction(testFilename, toArrMtd, myClassifer)
