
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os


class CIFAR10(Dataset):

    def __init__(self, val_size, noise, mode = "train", transform=None):
        self.mode = mode
        self.noise = noise
        self.transform = transform
        self.ToTensor = transforms.ToTensor()
        self.tt = transforms.ToPILImage()

        if self.mode == 'train' or self.mode == 'valid':

            self.cifar10 = datasets.CIFAR10('~/dataset', train=True, download=True)
            

            data_source = self.cifar10.data
            label_source = self.cifar10.targets
            label_source = np.array(label_source)

            self.data = []
            self.labels = []
            classes = range(10)

            ## training data
            if self.mode == 'train':
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[0:5000-val_size]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[0:5000-val_size]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

            elif self.mode == 'valid': ## validation data

                classes = range(10)
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[5000-val_size:5000]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[5000-val_size:5000]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

        elif self.mode == 'test':
            self.cifar10 = datasets.CIFAR10('~/dataset', train=False, download=True)
            self.data = self.cifar10.data
            self.labels = self.cifar10.targets


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.ToTensor(self.data[index])
        if self.mode == "train":
            #img = self.data[index]
            img = img + self.noise * torch.randn(img.size())
            img = torch.clamp(img, min=0.0, max=1.0)
            # img = (1-self.noise) * img + self.noise * torch.randn(img.size())
            #img = img + np.random.normal(0, self.noise, img.shape)

        target =  self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.tt(img)

        if self.transform:
            img = self.transform(img)

        return img, target



def get_cifar10_loader(valid_size_per_class, batch_size, transform_train=None):


    """Build and return data loader."""
    if transform_train == None:
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])

    transform_valid = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    dataset1 = CIFAR10(valid_size_per_class, 0.0, mode = "train", transform=transform_train)
    dataset2 = CIFAR10(valid_size_per_class, 0.0, mode = "valid", transform=transform_valid)
    dataset3 = CIFAR10(valid_size_per_class, 0.0, mode = "test", transform=transform_valid)

    train_loader = DataLoader(dataset=dataset1,
                             batch_size=batch_size,
                             shuffle=True)

    valid_loader = DataLoader(dataset=dataset2,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=dataset3,
                              batch_size=batch_size,
                              shuffle=True)
    return train_loader, valid_loader, test_loader



class GenDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        # stuff
        self.path = data_dict
        self.transform = transform
        self.label = torch.load(self.path + "label.pt", map_location = torch.device('cpu'))
        self.length = self.label.shape[0]
        self.tt = transforms.ToPILImage()
        
    def __getitem__(self, index, batch_size=128):
        # stuff
        batch_idx = index // batch_size
        img_batch = torch.load(self.path+"images/batch"+str(batch_idx)+".pt", map_location="cpu")
        img = img_batch[index % batch_size].squeeze()
        img = self.tt(img)
        label = self.label[index]

        if self.transform:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return self.length # of how many examples(images?) you have


class CompareDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        # stuff
        self.path = data_dict
        self.transform = transform
        self.label = torch.load(self.path + "label.pt", map_location = torch.device('cpu'))
        self.length = self.label.shape[0]
        self.tt = transforms.ToPILImage()
        
    def __getitem__(self, index, batch_size=128):
        # stuff
        batch_idx = index // batch_size
        img_batch = torch.load(self.path+"images/batch"+str(batch_idx)+".pt", map_location="cpu")
        img = img_batch[index % batch_size].squeeze()
        org_batch = torch.load(self.path+"images/batch"+str(batch_idx)+"_org.pt", map_location="cpu")
        org = org_batch[index % batch_size].squeeze()
        img = self.tt(img)
        org = self.tt(org)
        label = self.label[index]

        if self.transform:
            img = self.transform(img)
            org = self.transform(org)

        return (img, org, label)

    def __len__(self):
        return self.length # of how many examples(images?) you have


class MixupDataset(Dataset):
    def __init__(self, alpha, valid_size_per_class, transform_train=None):
    # stuff
        transform_valid = transforms.Compose([transforms.ToTensor(),])

        self.train_dataset = CIFAR10(valid_size_per_class, 0.0, mode = "train", transform=transform_valid)
        self.val_dataset = CIFAR10(valid_size_per_class, 0.0, mode = "valid", transform=transform_valid)
        self.alpha = alpha
        self.transform = transform_train

        self.length = len(self.train_dataset)
        self.tt = transforms.ToPILImage()

    def __getitem__(self, index):
        data, label = self.train_dataset[index]
        mix_label = (torch.randint(1,10,(1,) ) + label) % 10

        # import ipdb; ipdb.set_trace()

        data_val, _ = self.val_dataset[torch.randint(mix_label.item()*50,\
                (mix_label.item()+1)*50, (1,))]
        
        # import ipdb; ipdb.set_trace()

        data_mix = (1- self.alpha) * data + self.alpha * data_val

        if self.transform:
            data_mix = self.transform(self.tt(data_mix))

        return data_mix, label
        
    def __len__(self):
        return self.length


class MixupDatasetVerify(MixupDataset):
    def __init__(self, alpha, valid_size_per_class, transform_train=None):
        super().__init__(alpha, valid_size_per_class, transform_train=transform_train)
    
    def __getitem__(self, index):
        data_org, label = self.train_dataset[index]
        if self.transform:
            data_org = self.transform(self.tt(data_org))
        data_mix, label = super().__getitem__(index)
        return data_mix, data_org, label


class GaussianDatasetVerify(CIFAR10):
    def __init__(self, val_size, noise, mode="train", transform=None):
        super().__init__(val_size, noise, mode=mode, transform=transform)
    
    def __getitem__(self, index):
        data_org = self.data[index]
        if self.transform:
            data_org = self.transform(self.tt(data_org))
        data_noise, label = super().__getitem__(index)
        return data_noise, data_org, label


def load_masked(resultpath):
    batch_idx = 0
    img = []
    while os.path.isfile(resultpath+"images/batch"+str(batch_idx)+".pt"):
        img_batch = torch.load(resultpath+"images/batch"+str(batch_idx)+".pt", map_location="cpu")
        img.append(img_batch)
        batch_idx += 1
    img = torch.vstack(img)
    label = torch.load(resultpath + "label.pt", map_location = torch.device('cpu'))
    return img,label
    

class WholeGenDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        # stuff
        self.path = data_dict
        self.transform = transform

        img, label = load_masked(self.path)
        self.img = img
        self.label = label
        self.length = self.label.shape[0]
        self.tt = transforms.ToPILImage()
        
    def __getitem__(self, index):
        # stuff
        img = self.img[index].squeeze()
        img = self.tt(img)
        label = self.label[index]

        if self.transform:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return self.length # of how many examples(images?) you have


if __name__=="__main__":
    np.random.seed(0)

    tt = transforms.ToPILImage()

    transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    # dataset1 = CIFAR10(50, 0.4, "train", transform)
    
    # # dataset1 = MixupDataset(0.5, 50, transform_train=transform)

    # plt.figure(figsize=(12, 8))
    # for i in range(10):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(tt(dataset1[i+10][0]))
    #     plt.title("image=%d" % (i))
    #     plt.axis('off')
        
    # plt.show()

    # dataset2 = WholeGenDataset("./results0/", transform)
    # plt.figure(figsize=(12, 8))
    # for i in range(10):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(tt(dataset2[i+10][0]))
    #     plt.title("image=%d" % (i))
    #     plt.axis('off')
        
    # plt.show()
    svhn_train = datasets.SVHN('~/dataset', split='train',transform = transform, download=True)
    svhn_test = datasets.SVHN('~/dataset', split='test',transform = transform, download=True)
    svhn_extra = datasets.SVHN('~/dataset', split='extra',transform = transform, download=True)
    sub_idx = list(range(0, len(svhn_extra), 8))
    svhn_extra_sub = torch.utils.data.Subset(svhn_extra, sub_idx)
    # cifar10 = datasets.CIFAR10('~/dataset', train=True, download=True)
    import ipdb; ipdb.set_trace()

    # img, label = load_masked("./results0/")
