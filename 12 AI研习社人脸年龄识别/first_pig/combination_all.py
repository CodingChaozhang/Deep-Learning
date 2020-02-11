import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tqdm import tqdm
from keras.callbacks import *
import itertools
import torch


# 文件功能: 将给定的所有模型进行组合(从两两组合到最后全部组合),并依次输出准确率和mae到all_fusion.txt文件

output_dir = "models_val_outputs/"
txt_list = os.listdir(output_dir)

# 保存验证集val的年龄信息
with open('dataset/val_imglist.txt') as f:
    lines = f.readlines()
    age = []
    for line in tqdm(lines):
        age.append(line[:-1].split(' ')[1])


all_fusion = open("all_fusion.txt", 'a')

# 一个模型的信息也输出了,验证一下,(之前没有计算mae,这里正好可以计算一下)
for i in tqdm(range(1, len(txt_list) + 1)):
    for combination in tqdm(itertools.combinations(txt_list, i)):
        # print("models: ", combination)
        all_fusion.write("models: " + str(combination) + "\n")
        # 下面的步骤和test_to_csv.py文件中一样,融合模型的输出结果作为最终结果并比较
        txts = []
        num_line = len(age)
        prediction = 0
        for txt in combination:
            txt_name = os.path.join(output_dir, txt)
            prediction += np.loadtxt(txt_name)

        prediction = np.argmax(prediction, axis=1)
        age = np.array(age).astype(int)
        num_correct = np.sum(prediction == age)
        mae = np.sum(abs(prediction - age))

        # print("val_acc: ", num_correct / num_line, " ", "mae: ", mae / num_line, "\n")
        all_fusion.write("val_acc: " + str(num_correct / num_line) + "\tmae: " + str(mae / num_line) + "\n\n")

all_fusion.close()













