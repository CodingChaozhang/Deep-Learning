import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tqdm import tqdm
from keras.callbacks import *
import itertools


# 文件功能: 将给定的所有模型进行组合(从两两组合到最后全部组合),并依次输出准确率和mae到keras_fusion.txt文件


model_dir = "keras_models/"
model_list = os.listdir(model_dir)
output_dir = "models_val_outputs/"

# 保存验证集val的年龄信息
with open('dataset/val_imglist.txt') as f:
    lines = f.readlines()
    age = []
    for line in tqdm(lines):
        age.append(line[:-1].split(' ')[1])


keras_fusion = open("keras_fusion_2.txt", 'a')

# 一个模型的信息也输出了,验证一下,(之前没有计算mae,这里正好可以计算一下)
for i in range(1, len(model_list) + 1):
    for combination in itertools.combinations(model_list, i):
        # print("models: ", combination)
        keras_fusion.write("models: " + str(combination) + "\n")
        # 下面的步骤和keras_test.py文件中一样,融合模型的输出结果作为最终结果并比较
        txts = []
        for model in combination:
            txt_name = os.path.join(output_dir, '.'.join(model.strip().split('.')[0:-1]) + '.txt')
            txts.append(open(txt_name).readlines())
        num_line = len(txts[0])
        # print("num_line: ", num_line)
        num_correct = 0.0
        prediction = 0
        mae = 0.0
        for j in range(num_line):
            prediction = 0
            for txt in txts:
                prediction += np.array(txt[j].strip().split(' ')).astype(float)
            pred = np.argmax(prediction)
            num_correct += (int(pred) == int(age[j]))
            mae += abs(int(pred) - int(age[j]))

        # print("val_acc: ", num_correct / num_line, " ", "mae: ", mae / num_line, "\n")
        keras_fusion.write("val_acc: " + str(num_correct / num_line) + "\tmae: " + str(mae / num_line) + "\n\n")

keras_fusion.close()













