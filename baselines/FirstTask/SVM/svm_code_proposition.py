import pandas as pd
import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.utils import shuffle
from sklearn import metrics
from numpy import *


##累加计算句向量
def senvec(str):
    b = np.zeros(300, )
    val = str
    for i in val:
        j = w2v['i']
        b = np.add(b, j)
    return b
##形成句向量矩阵
def createmat(test):

    ##特征矩阵
    a = mat(zeros((len(test),300)))

    #句子
    k = 0
    for i in test:
        a[k] = senvec(i)
        k = k + 1

    return a

##模型训练
def create_model(d_train, d_test):
    print("训练样本 = %d" % len(d_train))
    print("测试样本 = %d" % len(d_test))

    #特征矩阵
    features = createmat(d_train.content)
    test_features = createmat(d_test.content)
    print("训练集特征向量shape ", features.shape)
    print("测试集特征向量shape ", test_features.shape)

    # svm模型
    svmmodel = SVC(C=1.0, kernel="linear")
    nn = svmmodel.fit(features, d_train.judge)
    print(nn)
    predict = svmmodel.score(test_features ,d_test.judge)
    print(predict)
    pre_test = svmmodel.predict(test_features)
    d_test["01category"] = pre_test
    d_test.to_excel("result.xlsx", index=False)
    #print(metrics.accuracy_score(d_test['01category'], d_test['judge']))##准确率



## 训练集预处理

#打开文件
train_text = pd.read_csv('train.txt',sep='\t',names = ['class','content','judge']) # 训练
test_text = pd.read_csv('test.txt',sep='\t',names = ['class','content','judge'])  #测试

d_train = train_text
d_test = test_text  # 测试

##打开词向量并创建字典
w2v = gensim.models.KeyedVectors.load_word2vec_format("sgns.baidubaike.bigram-char")

##训练
print("对样本进行01预测")
create_model(d_train, d_test)


