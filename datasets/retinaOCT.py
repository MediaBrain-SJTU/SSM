import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def load_data_OCT(datapath,batch_size):
    data_path = datapath+'/train/'
    test_data_path = datapath + '/test/'

    img_transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(root=data_path, transform=img_transform)
    test_dataset = ImageFolder(root=test_data_path, transform=img_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, test_dataloader