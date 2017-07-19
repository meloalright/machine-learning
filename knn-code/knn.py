import numpy as np
from sklearn import neighbors, datasets
import matplotlib.pyplot as pl

"""
 * 初始化一个digit类
 * @param {string}
 * @return {string}
"""
class digit(): 
    pass


"""
 * 创建数据集.
 * @param {string}
 * @return {string}
"""
digits = digit()
digits.data = [[0, 1],[1, 1], [2, 2], [3, 3], [4, 3], [5,10], [6,11], [7, 11], [8,11], [9, 10], [10, 10], [11, 10], [1, 2]]
digits.target = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]

"""
 * 选定训练集为前10个
 * @param {string}
 * @return {string}
"""
trainNum = 10
trainX = digits.data[0 : trainNum]
trainY = digits.target[0 : trainNum]

"""
 * 选定测试集为后3个
 * @param {string}
 * @return {string}
"""
testX = digits.data[trainNum: ]
testY = digits.target[trainNum: ]


"""
 * 选定knn阶数
 * @param {string}
 * @return {string}
"""
n_neighbors = 5
print('{K}NN训练'.format(K=n_neighbors))


"""
 * 预测结果
 * @param {string}
 * @return {string}
"""
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(trainX, trainY)
Z = clf.predict(testX)

"""
 * 打印结果
 * @param {string}
 * @return {string}
"""
print('预测结果:\n')
print(Z)

"""
 * 打印错误率
 * @param {string}
 * @return {string}
"""
print("\nthe total error rate is: %f" % ( 1 - np.sum(Z==testY) / float(len(testX)) ))



"""
 * 绘制示意图
 * @param {string}
 * @return {string}
"""
#训练数据
one_train_data = [trainX[index] for index in range(0, trainNum) if trainY[index] == 1]
zero_train_data = [trainX[index] for index in range(0, trainNum) if trainY[index] == 0]
#预测数据
one_predict_data = [testX[index] for index in range(0, 3) if Z[index] == 1]
zero_predict_data = [testX[index] for index in range(0, 3) if Z[index] == 0]
pl.plot([node[0] for node in one_train_data], [node[1] for node in one_train_data], 'ko')
pl.plot([node[0] for node in zero_train_data], [node[1] for node in zero_train_data], 'wo')
pl.plot([node[0] for node in one_predict_data], [node[1] for node in one_predict_data], 'bo')
pl.plot([node[0] for node in zero_predict_data], [node[1] for node in zero_predict_data], 'ro')
pl.xlabel('x')
pl.ylabel('y')
pl.xlim(0, 15)
pl.ylim(0, 15)
pl.show()
