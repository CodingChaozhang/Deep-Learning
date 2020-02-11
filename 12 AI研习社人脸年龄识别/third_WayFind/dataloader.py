from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class FADataset(Dataset):
    def __init__(self, filelist, isTrain=False):
        self.images = [item.strip().split(' ')[0] for item in open(filelist, 'r').readlines()]
        self.labels = [int(item.strip().split(' ')[1]) for item in open(filelist, 'r').readlines()]

        if isTrain:
            self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]) 
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])                       

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image = Image.open(self.images[index])
        image = Image.open(self.images[index]).convert('L')
        img = self.transform(image)
        label = self.labels[index]
        
        return img, label
