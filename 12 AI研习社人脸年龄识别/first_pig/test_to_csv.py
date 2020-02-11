import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tqdm import tqdm
from keras.callbacks import *


# 文件功能: 输出测试集test的csv提交文件(增加了验证集val的验证结果,确保逻辑正确)

# 存储测试集test的图片名称,以便写入csv文件
with open('dataset/test_imglist.txt') as f:
    lines = f.readlines()
    img_name = []
    for line in tqdm(lines):
        img_name.append(line[:-1].split('/')[1].split('.')[0])


# 存储验证集val中图片的年龄信息,以便计算准确率和mae
with open('dataset/val_imglist.txt') as f:
    lines = f.readlines()
    age = []
    for line in tqdm(lines):
        age.append(line[:-1].split(' ')[1])


model_list = []

# 加入融合的模型结果

# models: ('densenet121_0.2455_smt0.8.hdf5', 'densenet121_0.2397_baseline.hdf5',
# 'densenet121_0.2560_smt0.448.hdf5', 'densenet121_0.2474_smt0.655.hdf5')
# val_acc: 0.2521489971346705	mae: 3.961795606494747

model_list.append("densenet121_0.2397_baseline.txt")
model_list.append("densenet121_0.2455_smt0.8.txt")
model_list.append("densenet121_0.2474_smt0.655.txt")
model_list.append("densenet121_0.2560_smt0.448.txt")
# model_list.append("densenet121_0.2569_3.7619_0.517.txt")
# model_list.append("densenet_epoch8_val_acc0.2369.txt")
# model_list.append("densenet_epoch8_val_acc0.2426_0.001_64.txt")

# model_list.append("resnet_e3_acc0.2474_mae4.4441.txt")
# model_list.append("resnet_e12_acc0.2455_mae4.5377.txt")
# model_list.append("resnet_epoch0_val_acc0.2426.txt")
# model_list.append("resnet_epoch0_val_acc0.2483_0.00001j_64.txt")
# model_list.append("resnet_epoch1_val_acc0.2531_0.0001j_64.txt")
# model_list.append("resnet_epoch6_val_acc0.2426_0.001_64.txt")
# model_list.append("resnet_epoch4_val_acc0.2474.txt")
# model_list.append("resnet_epoch11_val_acc0.2436_0.001_64.txt")
# model_list.append("resnet_epoch14_val_acc0.2416.txt")
# model_list.append("mobilenet_epoch11_val_acc0.2474.txt")

##############################################################################################
# 验证集val准确率验证
load_dir_val = "models_val_outputs/"
txts_val = []
# 读取每个模型在测试集上的输出结果txt文件
num_val = len(age)
print("num_val: ", num_val)
num_correct = 0.0
mae = 0.0

prediction = 0
for txt in model_list:
    txt_name = os.path.join(load_dir_val, txt)
    prediction += np.loadtxt(txt_name)

prediction = np.argmax(prediction, axis=1)
age = np.array(age).astype(int)
num_correct = np.sum(prediction == age)
mae = np.sum(abs(prediction - age))

print("val_acc: ", num_correct / num_val, " ", "mae: ", mae / num_val)

###############################################################################################
# 测试集test输出结果到csv文件
load_dir_test = "models_test_outputs/"
txts_test = []
# 与测试集操作类似,只是没有真实结果,只将预测结果写入文件
num_test = len(img_name)
print("num_test: ", num_test)
output_txt = open('output.csv', 'a')

prediction = 0
for txt in model_list:
    txt_name = os.path.join(load_dir_test, txt)
    prediction += np.loadtxt(txt_name)

pred = np.argmax(prediction, axis=1)

for i in range(num_test):
    age = "%03d" % (int(pred[i]) + 1)
    text = img_name[i] + ',' + age + '\n'
    output_txt.write(text)
output_txt.close()

print("all done")
