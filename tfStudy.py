import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

np.random.seed(1)


def linearFunction():
    np.random.seed(1)

    # X = tf.constant(np.random.randn(3, 1), name="X")  # 定义一个维度是(3, 1)的常量，randn函数会生成随机数
    # W = tf.constant(np.random.randn(4, 3), name="W")
    # b = tf.constant(np.random.randn(4, 1), name="b")
    # Y = tf.add(tf.matmul(W, X), b)  # tf.matmul函数会执行矩阵运算

    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul函数会执行矩阵运算
    # Y = tf.add(tf.matmul(X ,W), b)
    # print(X.shape[0],X.shape[1])

    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    return result

def sigmoid(z):
    x = tf.placeholder(tf.float32,name="x")
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess :
        result = sess.run(sigmoid,feed_dict={x:z})

    return result

def cost(z_in,y_in):
    z = tf.placeholder(tf.float32,name="z")
    y = tf.placeholder(tf.float32,name="y")

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)

    sess = tf.Session()
    cost = sess.run(cost,feed_dict={z: z_in,y: y_in})
    sess.close()
    return cost

print("result = " + str(linearFunction()))
print("sigmoid(0) = "+str(sigmoid(0)))
print("sigmoid(12) = "+str(sigmoid(12)))

logits = np.array([0.2,0.4,0.7,0.9])
cost = cost(logits,np.array([0,0,1,1]))
print("cost = " + str(cost))

def one_hot_matrix(labels,C_in):
    C=tf.constant(C_in,name='C')
    one_hot_matrix = tf.one_hot(indices=labels,depth=C,axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

lables = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(lables,C_in=4)
print("one_hot = "+str(one_hot))

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    return ones

print("ones = "+ str(ones([3])))