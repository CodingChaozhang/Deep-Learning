import argparse
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from net import Net
from tqdm import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument("epoch", type=str, help="test epoch")
# args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]), ])

net = Net().to(device)
epoch = 32
net.load_state_dict(torch.load('./checkpoints/' + epoch + '.pkl'))
net.eval()

root = 'data/test/'

for item in tqdm(os.listdir(root)):
    img = Image.open(root + item).convert('L')
    x_data = Transform(img).float().unsqueeze(0).to(device)
    outputs = net(x_data)
    prediction = torch.max(F.softmax(outputs, dim=1), dim=1)[1].cpu().numpy()[0]
    age = prediction + 1
    if len(str(age)) == 1:
        result = '00' + str(age)
    else:
        result = '0' + str(age)
    with open('submission.csv', 'a+') as f:
        f.write(item.split('.')[0] + ',' + result + '\n')
