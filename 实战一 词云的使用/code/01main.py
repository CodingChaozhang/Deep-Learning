# _*_ coding:utf-8 _*_

# 进行英文词云生成
from wordcloud import WordCloud
import matplotlib.pyplot as plt

path = 'D://BaiduNetdiskDownload//深度有趣人工智能//实战一 词云的使用//'

# 打开文本
text = open(path + 'source//constitution.txt').read()
# 生成对象
wc = WordCloud().generate(text)

# 显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file(path + 'create_images//wordcloud1.png')
