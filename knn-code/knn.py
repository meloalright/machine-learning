import numpy as np
from sklearn import neighbors, datasets
import matplotlib.pyplot as pl

"""
 * 初始化一个digit类
 * @param {none}
 * @return {none}
"""
class digit(): 
    pass


"""
 * 创建数据集.
 * @param {none}
 * @return {none}
"""
digits = digit()
digits.data = [[0, 1],[1, 1], [2, 2], [3, 3], [4, 3], [5,10], [6,11], [7, 11], [8,11], [9, 10], [10, 10], [11, 10], [1, 2], [1 ,3], [1, 4], [2 ,0], [3, 3], [12, 12], [10, 10], [5, 7]]
digits.target = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0 ,0 ,0]

"""
 * 选定训练集为前10个
 * @param {none}
 * @return {none}
"""
trainNum = 10
trainX = digits.data[0 : trainNum]
trainY = digits.target[0 : trainNum]

"""
 * 选定测试集为后10个
 * @param {none}
 * @return {none}
"""
testX = digits.data[trainNum: ]
testY = digits.target[trainNum: ]


"""
 * 选定knn阶数
 * @param {none}
 * @return {none}
"""
n_neighbors = 5
print('{K}NN训练'.format(K=n_neighbors))


"""
 * 预测结果
 * @param {none}
 * @return {none}
"""
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(trainX, trainY)
Z = clf.predict(testX)

"""
 * 打印结果
 * @param {none}
 * @return {none}
"""
print('预测结果:\n')
print(Z)

"""
 * 打印错误率
 * @param {none}
 * @return {none}
"""
print("\nthe total error rate is: %f" % ( 1 - np.sum(Z==testY) / float(len(testX)) ))



"""
 * 绘制示意图
 * @param {none}
 * @return {none}
"""
#训练数据
one_train_data = [trainX[index] for index in range(0, trainNum) if trainY[index] == 1]
zero_train_data = [trainX[index] for index in range(0, trainNum) if trainY[index] == 0]
#预测数据
one_predict_data = [testX[index] for index in range(0, 10) if Z[index] == 1]
zero_predict_data = [testX[index] for index in range(0, 10) if Z[index] == 0]
pl.plot([node[0] for node in one_train_data], [node[1] for node in one_train_data], 'ko')
pl.plot([node[0] for node in zero_train_data], [node[1] for node in zero_train_data], 'wo')
pl.plot([node[0] for node in one_predict_data], [node[1] for node in one_predict_data], 'bo')
pl.plot([node[0] for node in zero_predict_data], [node[1] for node in zero_predict_data], 'ro')
pl.xlabel('x')
pl.ylabel('y')
pl.xlim(0, 15)
pl.ylim(0, 15)
pl.show()
