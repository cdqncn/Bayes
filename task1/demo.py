from numpy import *


def text_parse(bigString):  # input is big string, #output is word list
    '''
        接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    '''
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def create_vocab_list(dataSet):
    '''
		创建一个包含在所有文档中出现的不重复的词的列表。
	'''
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def words_to_vec(vocabList, inputSet):
    '''
		获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
	'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def cross_validation(X, Y, k):  # 使用K折交叉验证法提取数据
    num = int(len(X) / k)  # 50/5=10
    test_list = []
    test_label = []
    from random import randint
    for i in range(int(num / 2)):
        rand1 = randint(0, int(len(X) / 2))  # 在前半个数据集中抽取一部分作为测试集
        rand2 = randint(int(len(X) / 2) + 1, len(X) - 1)  # 在后半个数据集中抽取一部分作为测试集
        test_list.append(X[rand1])
        test_list.append(X[rand2])
        test_label.append(Y[rand1])
        test_label.append(Y[rand2])
        del (X[rand1])
        del (X[rand2])
        del (Y[rand1])
        del (Y[rand2])
    return X, Y, test_list, test_label  # 返回训练集、测试集


def main():
    words_list = []
    class_list = []
    import os
    for ham in os.listdir('ham'):
        with open('ham/' + ham, 'r', encoding='iso-8859-1') as f:
            words_list.append(text_parse(f.read()))
            class_list.append(0)  # 0表示正常词汇
    for spam in os.listdir('spam'):
        with open('spam/' + spam, 'r', encoding='iso-8859-1') as f:
            words_list.append(text_parse((f.read())))
            class_list.append(1)  # 1表示侮辱性词汇

    # 创建词汇表
    vocab_list = create_vocab_list(words_list)
    # 创建文档向量
    vec_list = [words_to_vec(vocab_list, words) for words in words_list]
    train_1, label_1, train_2, label_2 = cross_validation(vec_list, class_list, 5)  # 十折交叉验证法提取数据
    from sklearn.naive_bayes import MultinomialNB  # 使用多项式模型训练朴素贝叶斯分类器
    clf = MultinomialNB(alpha=2.0, class_prior=None, fit_prior=True)
    clf.fit(train_1, label_1)
    print("预测结果为" + str(clf.predict(train_2)))
    print('测试集合的真实标签为' + str(label_2))
    print('训练精度为：%%%.2f' % (clf.score(train_2, label_2) * 100))
    return


if __name__ == '__main__':
    main()
