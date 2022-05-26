"""
This function help to train model of different archtecture easily. Select model archtecture and training data, then output corresponding model.

"""
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from data_loader import *
import resnet
from grad_prune import GradPruneDefense

def train(model, data, device, maxepoch, data_path = './', transform = None, 
    save_per_epoch = 10, seed = 100, noise = 0.1, write_accuracy = False, args = None, 
    save_model=True, alpha = 0.0, random=False, defense='None'):
    """train.

    Parameters
    ----------
    model :
        model(option:'CNN', 'ResNet18', 'ResNet34', 'ResNet50', 'densenet', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
    data :
        data(option:'MNIST','CIFAR10')
    device :
        device(option:'cpu', 'cuda')
    maxepoch :
        training epoch
    data_path :
        data path(default = './')
    save_per_epoch :
        save_per_epoch(default = 10)
    seed :
        seed

    Examples
    --------
    >>>import deeprobust.image.netmodels.train_model as trainmodel
    >>>trainmodel.train('CNN', 'MNIST', 'cuda', 20)
    """

    torch.manual_seed(seed)

    train_loader, test_loader = feed_dataset(data, data_path, transform, noise, alpha)
    if data == 'CIFAR100':
        num_classes = 100
    else:
        num_classes = 10

    if (model == 'CNN'):
        import deeprobust.image.netmodels.CNN as MODEL
        #from deeprobust.image.netmodels.CNN import Net
        train_net = MODEL.Net().to(device)

    elif (model == "ResNet9"):
        import resnet as MODEL
        train_net = MODEL.ResNet9(random=random).to(device)
        learning_rate = 0.001
        
    elif (model == 'ResNet18' or model == 'ResNet18CIFAR100' or model == 'ResNet18SVHN'):
        #import deeprobust.image.netmodels.resnet as MODEL
        import resnet as MODEL
        train_net = MODEL.ResNet18(num_classes=num_classes, random=random).to(device)
        if data == 'CIFAR10_MASKED' or data == 'SVHN_MASKED':
            learning_rate = 0.05
        else:
            learning_rate = 0.1

    elif (model == 'ResNet34'):
        # import deeprobust.image.netmodels.resnet as MODEL
        import resnet as MODEL
        train_net = MODEL.ResNet34(random=random).to(device)
        learning_rate = 0.1

    elif (model == 'ResNet50'):
        # import deeprobust.image.netmodels.resnet as MODEL
        # train_net = MODEL.ResNet50().to(device)
        import resnet as MODEL
        train_net = MODEL.ResNet50(random=random).to(device)
        learning_rate = 0.1

    elif (model == 'densenet'):
        import deeprobust.image.netmodels.densenet as MODEL
        train_net = MODEL.densenet_cifar().to(device)

    elif (model == 'vgg11'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG11').to(device)
    elif (model == 'vgg13'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG13').to(device)
    elif (model == 'vgg16'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG16').to(device)
    elif (model == 'vgg19'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG19').to(device)


    optimizer = optim.SGD(train_net.parameters(), lr= learning_rate, momentum=0.9, weight_decay=5e-4)
    #scheduler = optim.lr_scpheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)
    if model == "ResNet9":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100], gamma=0.1)
    elif model == "ResNet18" or model == "ResNet34" or model == "ResNet50" or model=='ResNet18CIFAR100' or model == 'ResNet18SVHN':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)
    
    if defense == 'None':
        defense_method = None
    elif defense == 'GradPrune_05':
        defense_method = GradPruneDefense(0.5)
    elif defense == 'GradPrune_09':
        defense_method = GradPruneDefense(0.9)
    elif defense == 'GradPrune_095':
        defense_method = GradPruneDefense(0.95)
    elif defense == 'GradPrune_099':
        defense_method = GradPruneDefense(0.99)
    elif defense == 'GradPrune_0999':
        defense_method = GradPruneDefense(0.999)

    best_acc = 0.0
    best_epoch = 0
    best_model = None
    for epoch in range(1, maxepoch + 1):     ## 5 batches

        print(epoch)
        MODEL.train(train_net, device, train_loader, optimizer, epoch, defense_method)
        current_acc = MODEL.test(train_net, device, test_loader)

        if current_acc > best_acc:
            best_acc = current_acc
            best_epoch = epoch
            best_model = train_net

        if (save_model and (epoch % (save_per_epoch) == 0 or epoch == maxepoch)):
            if os.path.isdir('./trained_models/'):
                print('Save model.')
                torch.save(train_net.state_dict(), './trained_models/'+ data + "_" + model + "_epoch_" + str(epoch) + ".pt")
            else:
                os.mkdir('./trained_models/')
                print('Make directory and save model.')
                torch.save(train_net.state_dict(), './trained_models/'+ data + "_" + model + "_epoch_" + str(epoch) + ".pt")
                
        scheduler.step()

    print("Best Epoch: {}\tBest Acc:{:.3f}".format(best_epoch, best_acc))

    if args is not None:
        if os.path.isdir('./trained_models/'):
            print('Save last model.')
            torch.save(train_net.state_dict(), './trained_models/last_'+ data + "_" + model + defense +\
                "_noise" + str(args.noise) + "_alpha" + str(args.alpha) + "_results"+ args.result + ".pt")

            print('Save best model.')
            torch.save(best_model.state_dict(), './trained_models/best_'+ data + "_" + model + defense +\
                "_noise" + str(args.noise) + "_alpha" + str(args.alpha) + "_results"+ args.result + ".pt")
        else:
            os.mkdir('./trained_models/')
            print('Make directory and save last model.')
            torch.save(train_net.state_dict(), './trained_models/last_'+ data + "_" + model + defense +\
                "_noise" + str(args.noise) + "_alpha" + str(args.alpha) + "_results"+ args.result + ".pt")

            print('Make directory and save best model.')
            torch.save(best_model.state_dict(), './trained_models/best_'+ data + "_" + model + defense +\
                "_noise" + str(args.noise) + "_alpha" + str(args.alpha) + "_results"+ args.result + ".pt")
    
    if write_accuracy:
        test_write(train_net, device, test_loader, args)



def test_write(model, device, test_loader, args):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    with open('accuracy.txt', 'a') as file:
        if args is not None:
            file.write(' '.join(f'{k}={v}' for k, v in vars(args).items()) + '\n')

        file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        file.close()
        
    return test_loss,correct, len(test_loader.dataset)


def feed_dataset(data, data_dict, transform_cust, noise, alpha):
    if(data == 'CIFAR10'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=True, download = True,
                        transform=transform_train),
                 batch_size= 128, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True) #, **kwargs)

    
    elif(data== 'CIFAR10_MASKED'):
        if transform_cust == None:
            transform_cust = transforms.Compose([
                transforms.ToTensor(),
                ])
                
        # train_loader = torch.utils.data.DataLoader(
        #          GenDataset(data_dict, transform=transform_cust),
        #          batch_size= 128, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
                 WholeGenDataset(data_dict, transform=transform_cust),
                 batch_size= 128, shuffle=True)
        
        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        test_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10("~/dataset", train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True)
    
    elif(data == "CIFAR10_VAL"):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_loader = torch.utils.data.DataLoader(
                CIFAR10(50, 0.0, "train", transform_train),
                batch_size= 128, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                CIFAR10(50, 0.0, "test", transform_val),
                batch_size= 1000, shuffle=True)
        

    elif(data == 'MNIST'):
        train_loader = torch.utils.data.DataLoader(
                 datasets.MNIST(data_dict, train=True, download = True,
                 transform=transforms.Compose([transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])),
                 batch_size=128,
                 shuffle=True)

        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(data_dict, train=False, download = True,
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=1000,
                shuffle=True)
    
    
    elif(data == "CIFAR10_Out"):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        _, train_loader, test_loader = get_cifar10_loader(50,128, transform_train)
    
              
    elif(data == 'CIFAR10_Baseline'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                
        train_loader = torch.utils.data.DataLoader(
                 CIFAR10(50, noise, "train", transform_train),
                 batch_size= 128, shuffle=True)
        
        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        test_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10("~/dataset", train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True)
    

    elif(data == 'CIFAR10_Mixup_base'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                
        train_loader = torch.utils.data.DataLoader(
                 MixupDataset(alpha, 50, transform_train=transform_train),
                 batch_size= 128, shuffle=True)
        
        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        test_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10("~/dataset", train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True)
    
    elif(data == 'CIFAR100'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR100(data_dict, train=True, download = True,
                        transform=transform_train),
                 batch_size= 128, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR100(data_dict, train=False, download = True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True) #, **kwargs)
    
    elif(data == 'SVHN_EXTRA'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        svhn_extra = datasets.SVHN('~/dataset', split='extra',transform = transform_train, download=True)
        sub_idx = list(range(0, len(svhn_extra), 8))
        svhn_extra_sub = torch.utils.data.Subset(svhn_extra, sub_idx)
        train_loader = torch.utils.data.DataLoader(svhn_extra_sub, batch_size= 128, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.SVHN(data_dict, split='test', download=True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True) #, **kwargs)

    elif(data == 'SVHN_TRAIN'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        svhn_train = datasets.SVHN('~/dataset', split='train',transform = transform_train, download=True)
        svhn_train = torch.utils.data.Subset(svhn_train, list(range(50000)))

        train_loader = torch.utils.data.DataLoader(svhn_train ,batch_size= 128, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.SVHN(data_dict, split='test', download=True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True) #, **kwargs)
    
    elif(data== 'SVHN_MASKED'):
        if transform_cust == None:
            transform_cust = transforms.Compose([
                transforms.ToTensor(),
                ])
                
        # train_loader = torch.utils.data.DataLoader(
        #          GenDataset(data_dict, transform=transform_cust),
        #          batch_size= 128, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
                 WholeGenDataset(data_dict, transform=transform_cust),
                 batch_size= 128, shuffle=True)
        
        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        test_loader = torch.utils.data.DataLoader(
                 datasets.SVHN(data_dict, split='test', download=True,
                        transform=transform_val),
                batch_size= 1000, shuffle=True)


    elif(data == 'ImageNet'):
        pass

    return train_loader, test_loader


