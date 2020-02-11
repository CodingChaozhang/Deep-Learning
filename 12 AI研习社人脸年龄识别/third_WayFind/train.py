# -*- coding: utf-8 -*
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn.functional as F
from net import Net
from dataloader import FADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net()
# net.load_state_dict(torch.load("15.pkl"))
net.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
# optimizer = SGD(net.parameters(), lr=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, [8, 13], gamma=0.1, last_epoch=-1)
# scheduler = lr_scheduler.MultiStepLR(optimizer, [13, 19, 27, 33], gamma=0.1, last_epoch=-1)

best_accuracy = 0
best_loss = 10

dataset_train = FADataset('data/train.txt', True)
# dataset_train = FADataset('data/new_aug.txt', True)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)

EPOCHS = 15
# EPOCHS = 39

for epoch in range(EPOCHS):
    train_loss = 0.0
    net.train()
    for i, (imgs, target) in enumerate(train_loader):
        imgs = imgs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    for param_group in optimizer.param_groups:
        print('lr: %s' % param_group['lr'])
    print('Epoch: {}/{} - Average loss: {:.4f}'.format(epoch + 1, EPOCHS, train_loss))

    torch.save(net.state_dict(), 'checkpoints/' + str(epoch + 1) + '.pkl')
    scheduler.step(epoch + 1)
