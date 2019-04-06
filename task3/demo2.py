from sklearn.naive_bayes import MultinomialNB


def get_one_hot(original_data):
    X_data = [[0 for i in range(10000)] for j in range(len(original_data))]
    for i in range(len(original_data)):  # 取出第i行的数据
        for j in original_data[i]:
            X_data[i][int(j)] = 1
    return X_data


def main(k):  # k折交叉验证
    with open('train/train_data.txt', 'r', encoding='iso-8859-1') as f:
        X_train_data = get_one_hot([inst.strip().split(' ') for inst in f.readlines()])
    with open('train/train_labels.txt', 'r', encoding='iso-8859-1') as f:
        y_train_label = [inst.strip() for inst in f.readlines()]
        print(y_train_label)
    # 使用k折交叉验证法随机划分数据集
    X_test_data = []
    y_test_label = []
    from random import randint
    for i in range(0, int(len(X_train_data) / k)):
        rand = randint(0, len(X_train_data))
        X_test_data.append(X_train_data[rand])
        del (X_train_data[rand])
        y_test_label.append(y_train_label[rand])
        del (y_train_label[rand])
    bayes_clf = MultinomialNB(alpha=2.0, class_prior=None, fit_prior=True)  # 平滑处理
    bayes_clf.fit(X_train_data, y_train_label)
    print(bayes_clf.score(X_test_data, y_test_label))
    with open('test/test_data.txt', 'r', encoding='iso-8859-1') as fp:  # 读取测试数据文件
        X_test_data2 = get_one_hot([inst.strip().split(' ') for inst in fp.readlines()])
    with open('test/10.txt', 'w', encoding='iso-8859-1') as fp:  # 将测试结果写入一个新的文件中
        y_test_label2 = bayes_clf.predict(X_test_data2)
        for i in y_test_label2:
            fp.write(i + '\n')


if __name__ == '__main__':
    main(10)  # 十折交叉验证法
