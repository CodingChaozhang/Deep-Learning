import shutil
import os
import random

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("./")
    img_root = os.path.join(ROOT_DIR, "data/train/")
    img_moveto = os.path.join(ROOT_DIR, "data/val/")

    folder_list = os.listdir(img_root)
    folder_list.sort()
    for folder in folder_list:
        os.makedirs(os.path.join(img_moveto, folder))

    for folder in folder_list:
        img_train = os.path.join(img_root, folder)
        img_val = os.path.join(img_moveto, folder)
        if not os.path.isdir(img_train):
            continue
        img_list = os.listdir(img_train)
        random_list = random.sample(img_list, int(0.15 * len(img_list)))
        for img in random_list:
            shutil.move(os.path.join(img_train, img), os.path.join(img_val, img))

    # for folder in folder_list:
    #     img_train = os.path.join(img_root, folder)
    #     img_val = os.path.join(img_moveto, folder)
    #     if not os.path.isdir(img_val):
    #         continue
    #     img_list = os.listdir(img_val)
    #     for img in img_list:
    #         shutil.move(os.path.join(img_val, img), os.path.join(img_train, img))

    # 计算长度
    len_train = 0.0
    len_val = 0.0

    for folder in folder_list:
        # print("folder: ", folder, "\n")
        img_train = os.path.join(img_root, folder)
        if os.path.isdir(img_train):
            len_train += len(os.listdir(img_train))

        img_val = os.path.join(img_moveto, folder)
        if os.path.isdir(img_val):
            len_val += len(os.listdir(img_val))

    print("len_train: ", len_train, "len_val: ", len_val)
    print("all done")
