# 视频分解成图片
# 1 load加载视频 2 读取info 3 解码 单帧视频parse 4 展示 imshow
import cv2
# 获取一个视频打开cap
cap = cv2.VideoCapture('1.mp4')
# 判断是否打开
isOpened = cap.isOpened
print(isOpened)
#帧率
fps = cap.get(cv2.CAP_PROP_FPS)
#宽度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps,width,height)

i = 0
while(isOpened):
	if i == 100:
		break
	else:
		i = i+1
	(flag,frame) = cap.read() # 读取每一张 flag读取是否成功 frame内容
	fileName = 'imgs\\'+str(i) + '.jpg'
	print(fileName)
	if flag == True:
		#写入图片
		cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
print('end!')