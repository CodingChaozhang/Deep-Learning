# coding=utf-8
# author=yphacker

import os


def solve(base_dir):
    for dir in os.listdir(base_dir):
        path = '{}/{}'.format(base_dir, dir)
        new_dir = int((int(dir) - 1) / 10)
        new_path = 'data/train/{}'.format(new_dir)
        if not os.path.isdir(new_path):
            os.makedirs(new_path)
        os.system('cp -r {} {}'.format(path, new_path))


if __name__ == '__main__':
    solve('../../data/train')
