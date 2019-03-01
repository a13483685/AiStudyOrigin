#手把手写神经网络
import math

import matplotlib
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset,random_mini_batches,convert_to_one_hot,predict

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

index = 0
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:,index])))

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# 简单的归一化
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.
# one hot编码
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# print("total is :"+str(X_test))

def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X,Y

X,Y = create_placeholders(12288,6)
print("X = " + str(X))
print("Y = " + str(Y))

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2",[12,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }
    return parameters
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def forward_propagation(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # 计算第一层的z
    Z1 = tf.add(tf.matmul(W1, X), b1)
    # 在第一层的z上面执行relu激活操作，得到第一层的a。
    # 注意，tensorflow中已经帮我们实现了relu函数。
    # 之前是我们自己写不少python代码才能实现relu操作的。
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))

def computer_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

tf.reset_default_graph()
with tf.Session() as sess:
    X,Y = create_placeholders(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = computer_cost(Z3,Y)
    print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()  # 将计算图返回到默认空状态
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape  # (n_x: 特征数量, m : 训练集中的样本数)
    n_y = Y_train.shape[0]
    costs = []

    # 创建占位符
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 构建前向传播操作
    Z3 = forward_propagation(X, parameters)

    # 构建成本计算操作
    cost = computer_cost(Z3, Y)

    # 构建反向传播，为反向传播指定优化算法和学习率以及成本函数，这里我们使用adam算法，
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 定义初始化操作
    init = tf.global_variables_initializer()

    # 开始一个tensorflow的session
    with tf.Session() as sess:

        # 执行初始化操作
        sess.run(init)

        # 执行epochs指定的训练次数，一个epoch就是完整的向整个数据集学习一次
        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # 计算有多少个子训练集
            seed = seed + 1
            # 将数据集分成若干子训练集
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # 循环遍历每一个子训练集
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # 这行代码会使整个计算图被执行，从前向传播操作到反向传播操作，最后到参数更新操作。
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # 画出cost成本的走势图
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # 从计算图中获取训练好了的参数，后面我们就可以用这些参数来识别手语了！
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # 分别计算一下在训练集和测试集上面的预测精准度
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
parameters = model(X_train, Y_train, X_test, Y_test)


