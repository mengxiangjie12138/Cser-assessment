import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


########## Loader #########
def load_dataset(batchsize=50,data_dir = 'data'):
    # torchvision.transforms.Compose是用来管理所有transforms操作
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'test']}
    #print('image_datasets',image_datasets['train'])
    data_loaders = {x: data.DataLoader(image_datasets[x],
                                       batch_size=batchsize, shuffle=True)
                    for x in ['train', 'test']}
    #print('data_loaders', data_loaders)
    # data_loaders = {x : torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
    # 											num_workers=12, shuffle=True)
    # 					for x in ['train', 'val', 'test']}
    data_size = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return data_loaders, data_size

if __name__ == '__main__':

    data_loader,data_size=load_dataset(data_dir="E:\datasets\HIT_Parasite\DTGCN_data\\1_multistage_malaria_classification")

    print(len(data_loader['train']))
    i=0
    for data in data_loader['train']:
        i=i+1
        print(i)
        #inputs,labels=data
        #print(len(inputs),len(labels))
