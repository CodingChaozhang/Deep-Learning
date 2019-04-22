# 540 * 960 ==>64*128
import cv2

for i in range(0,100):
	fileName = 'imgs\\'+str(i+1)+'.jpg'
	print(fileName)
	img = cv2.imread(fileName)
	imgInfo = img.shape
	height = imgInfo[0]
	width = imgInfo[1]
	mode = imgInfo[2]
	dstHeight = 128
	dstWidth = 64
	dst = cv2.resize(img,(dstWidth,dstHeight))
	cv2.imwrite(fileName,dst)