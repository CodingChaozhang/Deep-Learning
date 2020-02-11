复现步骤：
用generate_train_val_file.py生成train.txt文件
将net.py中的预训练模型置为True：model = vgg19_bn(pretrained=True)
train文件：
optimizer = Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
scheduler = lr_scheduler.MultiStepLR(optimizer, [8, 13], gamma=0.1, last_epoch=-1)
dataset_train = FADataset('data/train.txt', True)
EPOCHS = 15
然后将上面的模型进行微调：
model = vgg19_bn(pretrained=False)
net.load_state_dict(torch.load("15.pkl"))
optimizer = SGD(net.parameters(), lr=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, [13, 19, 27, 33], gamma=0.1, last_epoch=-1)
dataset_train = FADataset('data/new_aug.txt', True)
EPOCHS = 39


一、数据分析
1、训练集非常不平衡。共70类，平均每类100张左右，但实际上超过均值的类别数只有不到30类。为了提高识别性能，我这边对均值以下的图像进行了随机两倍增强，数量在100-200之间的类别进行了随机1倍增强。增强方式有：gamma矫正、随机resize。
2、从整体数据集上看，各个类别的数据色彩参差不齐，为了避免干扰，将图像转为灰度输入训练。
3、输入图像分辨率：尝试了120 、 160 、 180 、200,200*200的性能最佳。
4、训练数据增强：数据都是对齐的，不适合进行更多的操作，只做了随机水平翻转。

二、BackBone
1、常用网络都试了一遍，VGG系列的性能最佳。
2、网络结构上没有更多的尝试，基本都在怎么解决过拟合。
3、测试结果的统计方式决定了回归网络不适合该任务，如SSRNet，预测结果都有一定的偏差。

三、调参
1、Warm up 、Focal loss 、 Label smoothing等都试了一遍，未见起色。
2、初始学习率设为0.001，先用Adam训练到收敛，再换成SGD训练，SGD的时候学习率要设小一点。
3、利用好自己每次提交的最佳模型，在其上面微调，往往会比之前的结果要好一点点。
4、模型选择：最小loss。

四、还没做的实验
1、按比例切不同尺度的image patch联合输入网络训练（参考C3AE）。
2、SE模块尝试过，没有深入。
3、多模型策略。

