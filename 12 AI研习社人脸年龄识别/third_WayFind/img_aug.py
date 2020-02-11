# coding:utf-8
import numpy as np
import cv2
import os
import random
from tqdm import tqdm


def gamma(img, gamma):
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_gamma = np.uint8(np.power((np.array(img) / 255.0), gamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)

    return image_gamma


def resize(img, size):
    img_resize = cv2.resize(img, size)
    img_resize = cv2.resize(img_resize, (200, 200))

    return img_resize


def img_process(img):
    if random.random() > 0.5:
        img = gamma(img, random.uniform(0.5, 1.9))
    if random.random() > 0.5:
        img = resize(img, (100, 100))
    if random.random() > 0.5:
        img = resize(img, (400, 400))
    if random.random() > 0.5:
        img = img[:, ::-1, :]

    return img


if __name__ == '__main__':
    root = 'data/train/'
    save = 'data/aug/'
    if not os.path.exists(save):
        os.makedirs(save)
    A = os.listdir(root)
    for item in tqdm(A):
        if len(item) > 3:
            continue
        B = os.listdir(root + item)
        if len(B) > 200:
            continue

        if not os.path.exists(save + item):
            os.makedirs(save + item)

        flag = 0
        for i in range(len(item)):
            if item[i] != '0':
                flag = i
                break
        label = int(item[flag:]) - 1

        for idx in range(len(B)):
            img = cv2.imread(root + item + '/' + B[idx])
            if len(B) < 100:
                img_a = img_process(img)
                cv2.imwrite(save + item + '/' + B[idx].split('.')[0] + 'a.jpg', img_a)
                with open('data/aug.txt', 'a+') as f:
                    f.write(save + item + '/' + B[idx].split('.')[0] + 'a.jpg ' + str(label) + '\n')
            img_p = img_process(img)
            cv2.imwrite(save + item + '/' + B[idx].split('.')[0] + 'p.jpg', img_p)
            with open('data/aug.txt', 'a+') as f:
                f.write(save + item + '/' + B[idx].split('.')[0] + 'p.jpg ' + str(label) + '\n')

                # cv2.imshow('1', img)
            # cv2.imshow('3', img)
            # cv2.imshow('2', cv2.imread(root + item + '/' + B[idx]))
            # cv2.waitKey(0)
