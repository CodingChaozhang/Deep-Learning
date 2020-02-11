#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


def generate(txt_name, dir, prefix, folder, label):
    files = os.listdir(dir)  # 里面的图片.png
    files.sort()

    listText = open(txt_name, 'a')
    for file in files:
        name = prefix + folder + '/' + file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()


train_path = 'data/train'  # 这里是你的图片的目录
val_path = 'data/val'  # 这里是你的图片的目录

if __name__ == '__main__':
    i = 0
    folderlist = os.listdir(train_path)  # 列举文件夹 001-070
    folderlist.sort()
    for folder in folderlist:
        if not os.path.isdir(os.path.join(train_path, folder)):
            continue
        generate('data/train_imglist.txt', os.path.join(train_path, folder), "train/", folder, i)
        generate('data/val_imglist.txt', os.path.join(val_path, folder), "val/", folder, i)
        i += 1
    print("all done")
