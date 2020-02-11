import os 

root = '../../data/train/'
A = os.listdir(root)

for item in A:
        if len(item) != 3:
                continue
        flag = 0
        for i in range(len(item)):
                if item[i] != '0':
                        flag = i
                        break
        label = int(item[flag:]) - 1
        B = os.listdir(root + item)

        for idx in B:
                with open('data/train.txt', 'a+') as f:
                        f.write(root + item + '/' + idx + ' ' + str(label) +'\n')

