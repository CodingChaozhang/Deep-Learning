# KNN 最近临域
# CNN 卷积神经网络

# 旧瓶装新酒：数字识别的不同
# 样本地址：http://yann.lecun.com/exdb/mnist/
# 代码下载
# from tensorflow.examples.tutorials.mnist import input_data  #第一次下载数据时用

# data = input_data.read_data_sets('MNIST_data/')

# Knn test 样本 K个 max
# 数据都是以随机数加载，四组数据，训练图片，训练标签，测试图片，测试标签。


# 1 load data
# 2 knn test train distance
# 3 knn k个最近图片 5张测试 500张训练图片做差 -->500张找出4张与当前最近的照片
# 4 解析图片中的内容parse centent==>label
# 5 label转化成具体的数字
# 6 检测结果是否正确
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

# load data 数据装载
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# 属性设置
trainNum = 55000
testNum = 10000
trainSize = 500
testSize = 5
k = 4
# 随机选取一定数量的测试图片与训练图片 在0-trainNum之间随机选取trainSize个数字，不可重复
trainIndex = np.random.choice(trainNum,trainSize,replace=False)
testIndex = np.random.choice(testNum,testSize,replace=False)

# 获取训练图片
trainData = mnist.train.images[trainIndex]
# 获取训练标签
trainLabel = mnist.train.labels[trainIndex]
# 获取测试图片
testData = mnist.test.images[testIndex]
# 获取测试标签
testLabel = mnist.test.labels[testIndex]

# 打印数据，获取数据的维度信息
print('trainData.shape',trainData.shape) #(500,784) 28*28=784图片所有像素点
print('trainLabel.shape',trainLabel.shape) #(500,10) 500行表示500个数，10列用来表示第几个数
print('testData.shape',testData.shape) #(5,784)
print('testLabel.shape',testLabel.shape) #(5,10)
print('testLabel',testLabel)

# TensorFlow input输入的定义
trainDataInput = tf.placeholder(shape=[None,784],dtype=tf.float32)
trainLabelInput = tf.placeholder(shape=[None,10],dtype=tf.float32)
testDataInput = tf.placeholder(shape=[None,784],dtype=tf.float32)
testLabelInput = tf.placeholder(shape=[None,10],dtype=tf.float32)

# KNN 距离 distance
# f1测试数据输入维度扩展 由5*784==>5*1*784
f1 = tf.expand_dims(testDataInput,1)
# f2测试图片与训练图片两者之差
f2 = tf.subtract(trainDataInput,f1)
# f3完成数据累加 784个像素点的差值
f3 = tf.reduce_sum(tf.abs(f2),reduction_indices=2)
# f4对f3取反
f4 = tf.negative(f3)
# f5,f6 选取f4中最大的4个数，即选取f3中最小的4个数（值的内容及值的下标）
f5,f6 = tf.nn.top_k(f4,k=4)
# f6 存储最近的四张图片的下标
f7 = tf.gather(trainLabelInput,f6)
# f8 将最近的四张图片 累加 维度为1
f8 = tf.reduce_sum(f7,reduction_indices=1)
# f9 根据训练数据 推测的值
f9 = tf.argmax(f8,dimension=1)


with tf.Session() as sess:
	p1 = sess.run(f1,feed_dict={testDataInput:testData[0:5]})
	print('p1=',p1.shape) #(5,1,784)
	p2 = sess.run(f2,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
	print('p2=',p2.shape) #(5,500,784)
	p3 =sess.run(f3,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
	print('p3=',p3.shape) #(5,500)
	print('p3[0,0]',p3[0,0])

	p4 = sess.run(f4,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
	print('p4[0,0]',p4[0,0])

	p5,p6 = sess.run((f5,f6),feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
	print('p5=',p5.shape)
	print('p6=',p6.shape)
	#p5=(5,4) 每一张测试图片（5张）分别对应当前四张最近的图片
	#p6=(5,4) 每一张测试图片（5张）分别对应当前四张最近的图片下标
	print('p5[0,0]=',p5[0])
	print('p6[0,0]=',p6[0])

	p7 = sess.run(f7,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
	print('p7=',p7.shape) #(5,4,10)
	print('p7[]=',p7)

	p8 = sess.run(f8,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
	print('p8=',p8.shape) #(5,10)
	print('p8[]=',p8)

	p9 = sess.run(f9,feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
	print('p9=',p9.shape)
	print('p9[]=',p9)

	# p10 根据测试数据 得到的值
	p10 = np.argmax(testLabel[0:5],axis=1)
	print('p10[]=',p10)

# 比较p9和p10,计算准确率
j = 0
for i in range(0,5):
	if p10[i] == p9[i]:
		j=j+1
print('ac=',j*100/5)

