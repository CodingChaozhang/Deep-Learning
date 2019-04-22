# _*_ coding:utf-8 _*_
# 词云的颜色从蒙版中抽取

from wordcloud import  WordCloud,ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

path = 'D://BaiduNetdiskDownload//深度有趣人工智能//实战一 词云的使用//'

# 打开文本
text = open(path+'source//xyj.txt',encoding='utf-8').read()

# 中文分词
text = ''.join(jieba.cut(text))
print(text[:100])

# 生成对象
mask = np.array(Image.open(path+"source//color_mask.png"))
wc = WordCloud(mask=mask,font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

# 从图片中生成颜色
image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file(path+"create_images//wordcloud5.png")