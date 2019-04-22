# CNN 卷积神经网络

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# 设置input
imageInput = tf.placeholder(tf.float32,[None,784]) #输入图片28*28=784维
labelInput = tf.placeholder(tf.float32,[None,10]) #标签 参考CNN内容

# 数据维度的调整 data reshape 由[None,784]-->M*28*28*1 即2维转换成4维
imageInputReshape = tf.reshape(imageInput,[-1,28,28,1]) #完成维度的调整

# 卷积运算 w0是权重矩阵 本质是卷积内核 5*5 output 32  一维 1 stddev 方差
w0 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b0 = tf.Variable(tf.constant(0.1,shape=[32]))

#layer1:激励函数+卷积运算
# imageInputReshape: M*28*28*1 w0:5,5,1,32 strides:步长 padding:卷积核可以停留到图像的边缘
layer1 = tf.nn.relu(tf.nn.conv2d(imageInputReshape,w0,strides=[1,1,1,1],padding='SAME')+b0)
# M*28*28*32 ===>M*7*7*32
# 池化层 下采样 数据量减少了很多
layer1_pool = tf.nn.max_pool(layer1,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')


#layer2(output输出层):softmax（激励函数+乘加运算）
w1 = tf.Variable(tf.truncated_normal([7*7*32,1024],stddev=0.1))
b1 = tf.Variable(tf.constant(0.1,shape=[1024]))
h_reshape = tf.reshape(layer1_pool,[-1,7*7*32]) #M*7*7*32->N*N1
# [N*7*7*32] [7*7*32,1024]=N*1024
h1 = tf.nn.relu(tf.matmul(h_reshape,w1)+b1)

# softmax
w2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b2 = tf.Variable(tf.constant(0.1,shape=[10]))
pred = tf.nn.softmax(tf.matmul(h1,w2)+b2) # N*1024 1024*10=N*10
# N*10(概率)
loss0 = labelInput*tf.log(pred)
loss1 = 0
for m in range(0,100):
	for n in range(0,10):
		loss1 = loss1 + loss0[m,n]
loss = loss1/100

# train 训练
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# run 运行
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100):
		images,labels = mnist.train.next_batch(500)
		sess.run(train,feed_dict={imageInput:images,labelInput:labels})

		pred_test = sess.run(pred,feed_dict={imageInput:mnist.test.images,labelInput:labels})
		acc = tf.equal(tf.arg_max(pred_test,1),tf.arg_max(mnist.test.labels,1))
		acc_float = tf.reduce_mean(tf.cast(acc,tf.float32))
		acc_result = sess.run(acc_float,feed_dict={imageInput:mnist.test.images,labelInput:mnist.test.labels})
		print(acc_result)
