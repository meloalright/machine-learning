`ps: [tf.nn.softmax] 输出的是多项分布 对于二值分类问题 Softmax输出两个值 这两个值相加为1`   
   
```
def softmax(x):
       return np.exp(x) / np.sum(np.exp(x), axis=0)
```   
   
`logistic-regression.py 逻辑回归`   
