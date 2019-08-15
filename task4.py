import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = np.load('F:/泰迪/task3/data.npy')
labels = np.load('F:/泰迪/task3/labels.npy')

labels = tf.one_hot(labels,depth=10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(labels)

data = data.reshape([5445,784])
labels = a.reshape([5445,10])

#划分训练测试
train_x,test_x,train_y,test_y = train_test_split(data,labels,test_size=0.2,random_state=0)

#定义批次大小，每次100张图片
batch_size = 100
batch_num = train_y.shape[0] // batch_size

#定义学习率
learning_rate = tf.Variable(0.0001,dtype=tf.float32)

#定义占位符，最后通过feed_dict={}传入
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
dropout_rate = tf.placeholder(tf.float32)

#正则化项权重
regularization_rate = 0.0001

#调整图片为二维
x_images = tf.reshape(x,[-1,28,28,1])

#定义权重函数
def get_weight(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
#定义偏置项函数
def get_bias(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
#定义卷积层函数
#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
#input:输入图片，格式为[batch,长,宽,通道数]，batch为批次大小，长宽就是图片的像素大小，通道数实际上是输入图片的三维矩阵的深度，灰度图为1，彩色图为3
#filter:卷积核，格式为[长,宽,输入通道数,输出通道数],长宽为卷积核大小,输入通道数和input通道数一致,输出通道数可以随意指定，其实也就是卷积核个数
#strides:卷积核移动步长,格式为[1,x,y,1]
#padding:卷积核在边缘处的处理方法,padding='SAME'为补0,padding='VALID'为不补0
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#定义池化层函数
#tf.nn.max_pool(input, ksize, strides, padding, name=None)
#input:需要池化的输入，格式为[batch,长,宽,通道数]，一般池化层接在卷积层后面，所以input通常是卷积后的结果
#ksize:池化窗口的大小，格式为[1,x,y,1]
#strides:池化窗口的移动步长，格式为[1,x,y,1]
def pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积池化
w1 = get_weight([5,5,1,32])
b1 = get_weight([32])
conv1 = tf.nn.relu(conv2d(x_images,w1)+b1)
pool1 = pooling(conv1)

#第二层卷积池化
w2 = get_weight([5,5,32,64])
b2 = get_bias([64])
conv2 = tf.nn.relu(conv2d(pool1,w2)+b2)
pool2 = pooling(conv2)

#图片大小转换，把图片换为一维
pool2_shape = pool2.get_shape()
nodes = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
flat = tf.reshape(pool2,[-1,nodes])

#第一层全连接层
w3 = get_weight([7*7*64,512])
b3 = get_bias([512])
layer1 = tf.nn.relu(tf.matmul(flat,w3)+b3)
layer1_dropout = tf.nn.dropout(layer1,dropout_rate)
#把要计算正则化的权重w3加入tf.GraphKeys.WEIGHTS集合
tf.add_to_collection(tf.GraphKeys.WEIGHTS,w3)

#第二层全连接层
w4 = get_weight([512,10])
b4 = get_bias([10])
prediction = tf.matmul(layer1_dropout,w4)+b4
#把要计算正则化的权重w4加入tf.GraphKeys.WEIGHTS集合
tf.add_to_collection(tf.GraphKeys.WEIGHTS,w4)

#定义L2正则化
regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
#计算全部正则化项，有两个参数，regularizer和weights_list，weights_list默认为GraphKeys.WEIGHTS中的weights
regularization = tf.contrib.layers.apply_regularization(regularizer)
#定义损失函数
cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#整合损失函数和正则化
loss = cross_entropy_mean + regularization
#定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate)
#训练的目标
train = optimizer.minimize(loss)

#计算正确率
correct = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

#初始化
init = tf.global_variables_initializer()

#保存模块
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for step in range(20):
        #学习率指数衰减
        sess.run(tf.assign(learning_rate,learning_rate*0.95**step))
        for i in range(batch_num):
            batch_x = train_x[i*batch_size:(i+1)*batch_size,:]
            batch_y = train_y[i*batch_size:(i+1)*batch_size,:]
            sess.run(train,feed_dict={x:batch_x,y:batch_y,dropout_rate:1.0})
        acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y,dropout_rate:1.0})
        print('Iter ' + str(step) + ' testing accuracy : ' + str(acc))
        #保存模型
        saver.save(sess,'F:/model/')
