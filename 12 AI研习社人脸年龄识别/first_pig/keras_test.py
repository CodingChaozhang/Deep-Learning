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

# 加入融合的模型(后来发现直接加入模型结果就可以了!在test_to_csv.py中改正了)
model_list.append("densenet121_0.2455_smt0.8.hdf5")
# model_list.append("densenet121_0.2474_smt0.655.hdf5")
model_list.append("densenet121_0.2397_baseline.hdf5")
model_list.append("densenet121_0.2560_smt0.448.hdf5")
# model_list.append("densenet121_0.2569_3.7619_0.517.hdf5")

##############################################################################################
load_dir_val = "models_val_outputs/"
txts_val = []
# 读取每个模型在测试集上的输出结果txt文件放入txts_val
for model in model_list:
    txt_name = os.path.join(load_dir_val, '.'.join(model.strip().split('.')[0:-1]) + '.txt')
    txts_val.append(open(txt_name).readlines())

num_line = len(txts_val[0])
print("num_line: ", num_line)
num_correct = 0.0
mae = 0.0

# 遍历输出结果的每一行(即每个测试数据) (后来发现直接用np.loadtxt函数就可以了!在test_to_csv.py中已改)
for i in range(num_line):
    prediction = 0
    # 遍历每个模型输出结果的第i行,相加得到融合后的结果,与正确结果相比较
    for txt in txts_val:
        prediction += np.array(txt[i].strip().split(' ')).astype(float)
    pred = np.argmax(prediction)
    num_correct += (int(pred) == int(age[i]))
    mae += abs(int(pred) - int(age[i]))

print("val_acc: ", num_correct / num_line, " ", "mae: ", mae / num_line)

###############################################################################################
load_dir_test = "models_test_outputs/"
txts_test = []
# 与测试集操作类似,只是没有真实结果,只将预测结果写入文件
for model in model_list:
    txt_name = os.path.join(load_dir_test, '.'.join(model.strip().split('.')[0:-1]) + '.txt')
    txts_test.append(open(txt_name).readlines())

num_line = len(txts_test[0])
print("num_line: ", num_line)

output_txt = open('keras_output.csv', 'a')

for i in range(num_line):
    prediction = 0
    for txt in txts_test:
        prediction += np.array(txt[i].strip().split(' ')).astype(float)
    pred = np.argmax(prediction)
    age = "%03d" % (int(pred) + 1)
    text = img_name[i] + ',' + age + '\n'
    output_txt.write(text)
output_txt.close()

print("all done")
