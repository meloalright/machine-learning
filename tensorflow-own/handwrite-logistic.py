'''
 meloalright
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


MNIST = mnist

mndata = mnist

sess = tf.Session()
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder("float", [None, 10])
# 使用Tensorflow自带的交叉熵函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess.run(init)

# 训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(500)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})




print('\n训练结束->开始预测选定的测试集')

# images = mndata.test.images
# labels = mndata.test.labels
# 选定测试集
arr = ['./A000.png', './A001.png', './A002.png', './A003.png', './A004.png', './A005.png']
for path in arr:
    image = Image.open('./diy_test_data/{path}'.format(path=path)).convert('L')

    print('\n打开{path}'.format(path=path))

    # 打印图片
    dmp = np.array(image).reshape(28, 28)
    plt.imshow(dmp, cmap='gray')
    plt.show()
    content = input("\n按回车键继续")

    # 测试新图片，并输出预测值
    a = np.array(image).reshape(1, 784)
    y = tf.nn.softmax(y)  # 为了打印出预测值，我们这里增加一步通过softmax函数处理后来输出一个向量
    result = sess.run(y, feed_dict={x: a})  # result是一个向量，通过索引来判断图片数字
    print('\n\n预测值为：')
    print(result)

    # 输出结果
    print('\n\n结果：')
    count = 0
    for i in result[0]:
        if i > 0.1 and i < 2.01:
            print(str(count))
        count += 1

    content = input("\n按回车键继续----------------")

