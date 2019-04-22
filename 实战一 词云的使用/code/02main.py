# _*_ coding:utf-8 _*_
# 中文不分词效果展示
from wordcloud import WordCloud
import matplotlib.pyplot as plt

path = 'D://BaiduNetdiskDownload//深度有趣人工智能//实战一 词云的使用//'

# 打开文本
text = open(path + 'source//xyj.txt',encoding = 'UTF-8').read()
# 生成对象
wc = WordCloud(font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

# 显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file(path + 'create_images//wordcloud2.png')