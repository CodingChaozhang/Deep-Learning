# 1 思想 分类器
# 2 如何？寻求一个最优的超平面 分类
# 3 核：line
# 4 数据：样本
# 5 训练：SVM_create train predict

# SVM本质 寻求一个最优的超平面 分类

# SVM  line直线
# 身高和体重的分类 训练  预测 男女

# SVM 所有的数据都要有label
# 有标签的训练 ====> 监督学习 0 负样本 1 正样本
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 准备data
rand1 = np.array([[155,48],[159,50],[164,53],[168,56],[172,60]])
rand2 = np.array([[152,53],[156,55],[160,56],[172,64],[176,65]])
data = np.vstack((rand1,rand2))
data = np.array(data,dtype='float32')

# 2 label 标签
label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

# 3 训练SVM
# m1 机器学习模块 SVM_create()创建向量机
svm = cv2.ml.SVM_create()
# 属性设置
svm.setType(cv2.ml.SVM_C_SVC) # svm type
svm.setKernel(cv2.ml.SVM_LINEAR) # line 线性分类器
svm.setC(0.01)
# 训练
svm.train(data,cv2.ml.ROW_SAMPLE,label)

# 4 完成数据的预测
pt_data = np.vstack([[167,55],[162,57]]) # 0 女生 1 男生
pt_data = np.array(pt_data,dtype='float32')
# 开始预测
(par1,par2) = svm.predict(pt_data)
print(par2) 
