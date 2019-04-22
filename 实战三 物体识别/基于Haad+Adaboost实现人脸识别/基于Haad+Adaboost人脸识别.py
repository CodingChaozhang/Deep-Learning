# haar
# 1.什么是haar特征？特征 = 像素 运算 得到的 某个结果（具体值 向量 矩阵 多维）
# 2.如何利用特征 区分目标？阈值判决
# 3.如何得到判决？ 机器学习得到判决
# 三个问题 1.特征 2.判决 3 得到判决

# Haar特征
# 特征=白色-黑色 特征=整个区域*权重+黑色*权重 特征=(p1-p2-p3+p4)*w
# 积分图 特征=(p1-p2-p3+p4)*w

# adaboost 分类器
# 1 分类器的结构 2 adaboost计算过程 3 xml文件结构
#  haar > T1 and haar > T2
# 整体流程 haar->Node z1 z2 z3 Z=sum(z1,z2,z3)
# Z>T  y1 y2 y3 弱分类器
# x = sum(y1,y2,y3) 强分类器
# x>T

# adaboost 训练
# 1 初始化数据权值分布
# 2 遍历阈值 误差概率p
# minp t
# 3 G1(x)
# 4 更新权值分布 update

# 程序步骤
# 1 load xml  2 load jpg 3 haar gray 4 detect 5 draw
import cv2
import numpy as np
# xml文件的引入
face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')

# load jpg
img = cv2.imread('face.jpg')
cv2.imshow('src',img)

# haar  gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# dect
# 1 人脸灰度图片数据 2 缩放系数 3 目标大小
faces = face_xml.detectMultiScale(gray,1.3,5)

print('face=',len(faces))
# draw
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	# 眼睛识别代码
	roi_gray = gray[y:y+h,x:x+w]
	roi_img = img[y:y+h,x:x+w]
	eyes = eye_xml.detectMultiScale(roi_gray)
	print('eyes=',len(eyes))
	for (e_x,e_y,e_w,e_h) in eyes:
		cv2.rectangle(roi_img,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)

cv2.imshow('dst',img)
cv2.waitKey(0)

