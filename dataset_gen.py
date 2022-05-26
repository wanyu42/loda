import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import resnet
import os
from PIL import Image
from data_loader import *


def test(model, device, test_loader):
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class ConvOutHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        self.output = output

    def close(self):
        self.hook.remove()


class BatchStatHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        
        mean = input[0].mean([0, 2, 3])
        # import ipdb; ipdb.set_trace()
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # import ipdb; ipdb.set_trace()
        self.mean = mean
        self.var = var

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        # r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
        #     module.running_mean.data.type(var.type()) - mean, 2)

        # self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()



def init_gen(data, valid_dataset, target, device, start_point, noise_level, alpha, swap_num):
    if start_point == "gaussian":
            data_r = (data.clone().detach() + noise_level * torch.randn(data.size()).to(device)).requires_grad_(True)

    elif start_point == "mixup":
        data_r = []
        out_class = (torch.randint(1,10,(target.shape[0],) ).to(device) + target) % 10

        for ii in range(target.shape[0]):
            data_val, _ = valid_dataset[torch.randint(out_class[ii].item()*50,\
                    (out_class[ii].item()+1)*50, (1,)).item()]
            data_mix = (1- alpha) * data[ii] + alpha * data_val.to(device)
            data_r.append(data_mix.to(device))
        data_r = torch.stack(data_r).requires_grad_(True)
    
    elif start_point == "svhn_extra":
        data_r = []
        # trans_valid = transforms.Compose([transforms.ToTensor(),])
        # valid_new = datasets.CIFAR100('~/dataset', train=True, download=True)
        r_idx = torch.randint(0, len(valid_dataset), (target.shape[0], ))
        for i in r_idx:
            data_r.append(valid_dataset[i][0])
        data_r = torch.stack(data_r).to(device).requires_grad_(True)
    
    elif start_point == "reproduce":
        data_r = data[torch.randperm(target.shape[0])]
        data_r = data_r.requires_grad_(True)
    
    elif start_point == "shift":
        s_idx = (torch.randint(1,target.shape[0],(1,) ).to(device) + torch.arange(target.shape[0],device=device)) % target.shape[0]
        data_r = data[s_idx]
        data_r = data_r.requires_grad_(True)

    elif start_point == 'repromixup':
        # import ipdb; ipdb.set_trace()
        data_r = []
        r_idx = torch.randint(0, len(valid_dataset), (target.shape[0], ))
        for i in r_idx:
            data_r.append(valid_dataset[i][0])
        data_r = torch.stack(data_r).to(device).requires_grad_(True)
    
    elif start_point == "cifar100":
        data_r = []
        trans_valid = transforms.Compose([transforms.ToTensor(),])
        valid_new = datasets.CIFAR100('~/dataset', train=True, download=True)
        r_idx = torch.randint(0, len(valid_new), (target.shape[0], ))
        for i in r_idx:
            data_r.append(trans_valid(valid_new[i][0]))
        data_r = torch.stack(data_r).to(device).requires_grad_(True)

    elif start_point == "permmixup":
        data_r = 0.5*data[torch.randperm(target.shape[0])] + 0.5*data[torch.randperm(target.shape[0])]
        data_r = data_r.requires_grad_(True)

    return data_r


def conv_loss(conv_out, conv_out_r, batch_size, weight_conv, inner):
    loss = 0
    for conv_layer_out, conv_layer_out_r in zip(conv_out, conv_out_r):
        loss += ((conv_layer_out - conv_layer_out_r)**2).sum() / batch_size
    # import ipdb; ipdb.set_trace()
    if inner % 100 == 0:
        print(f'conv loss {weight_conv * loss.item():4.2f}')

    return weight_conv * loss


def loss_feature_diff(model, data, data_r, target, inner, conv_out_layers, loss_type, conv, weight_conv, conv_part, feat_weight):
    with torch.no_grad():
        output_target = model(data).clone().detach()
        # feature_target = model.penultimate.clone().detach()
        # previous_target = model.previous.clone().detach()
        feature_target = model.penultimate.clone().detach()
        previous_target = model.previous.clone().detach()
        if conv:
            conv_out = [mod.output for mod in conv_out_layers]

    output_r = model(data_r)
    feature_r = model.penultimate
    previous_r = model.previous
    if conv:
        conv_out_r = [mod.output for mod in conv_out_layers]
    #import ipdb; ipdb.set_trace()

    if loss_type == "l2":
        feature_diff = 0                   
        feature_diff += torch.sum((feature_r - feature_target)**2) / target.shape[0]
        feature_diff += torch.sum((previous_r - previous_target)**2) / target.shape[0]
    elif loss_type == "kl":
        kl = torch.nn.KLDivLoss(reduction = "batchmean", log_target = True)
        feature_diff = kl(F.log_softmax(feature_r,dim=1), F.log_softmax(feature_target, dim=1))
    feature_diff *= feat_weight

    if inner % 100 == 0:
        print(f'feat loss {feature_diff.item():4.2f}')

    if conv:
        if conv_part == 'former':
            low = 0; high = 5
        elif conv_part == 'middle':
            low = -10; high = -5
        elif conv_part == 'latter':
            low = -15; high = -10
        elif conv_part == "whole":
            low = 0; high = 15
        feature_diff += conv_loss(conv_out[low:high], conv_out_r[low:high], target.shape[0], weight_conv, inner)

    return feature_diff


def pgd_step(loss_fn, data_r, target, step_size):
    grad = torch.autograd.grad(loss_fn, data_r,
                                    retain_graph=False, create_graph=False)[0]
    grad_norms = torch.norm(grad.view(target.shape[0], -1), p=2, dim=1)
    grad = step_size * grad/ ( grad_norms.view(target.shape[0], 1, 1, 1) + 1e-8 )
    # grad = 1 * grad/ ( grad_norms.view(target.shape[0], 1, 1, 1) + 1e-8 )
    # print(grad_norms)
    data_r = data_r.detach() - grad
    data_r = torch.clamp(data_r, min=0.0, max=1.0).detach()

    return data_r


def gd_step(loss_fn, data_r, target, step_size):
    grad = torch.autograd.grad(loss_fn, data_r,
                                    retain_graph=False, create_graph=False)[0]

    data_r = data_r.detach() - step_size * grad
    data_r = torch.clamp(data_r, min=0.0, max=1.0).detach()

    return data_r


def save_img(resultpath, data_r, data, idx):
    if os.path.isdir(resultpath+'images/'):
        # for im_idx in range(target.shape[0]):
        #     # tt(data_r[im_idx].cpu()).save(resultpath+"images/img"+str(idx*batch_size + im_idx)+".png")
        #     target_r.append(target[im_idx])
        torch.save(data_r, resultpath+"images/batch"+str(idx)+".pt")
        torch.save(data, resultpath+"images/batch"+str(idx)+"_org.pt")
    else:
        os.mkdir(resultpath)
        os.mkdir(resultpath+'images/')
        # for im_idx in range(target.shape[0]):
        #     # tt(data_r[im_idx].cpu()).save(resultpath+"images/img"+str(idx*batch_size + im_idx)+".png")
        #     target_r.append(target[im_idx])
        torch.save(data_r, resultpath+"images/batch"+str(idx)+".pt")
        torch.save(data, resultpath+"images/batch"+str(idx)+"_org.pt")


def generate_dataset(model, train_loader, valid_dataset, device, batch_size,resultpath, params,
        max_iterations=1000, inner_per_distance_record=10, inner_per_image=100):

    model.eval()
    model = model.to(device)
    tt = transforms.ToPILImage()

    conv_out_layers = []
    if params['conv']==True:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_out_layers.append(ConvOutHook(module))

    global_feature_diff_history = []
    global_image_diff_history = []
    global_loss_history = []
    target_r = []

    for idx, (data, target) in enumerate(train_loader):
        print("=================== img" + str(idx) + "=========================")
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        ########### Starting point Initilization ###############################################################
        
        data_r = init_gen(data, valid_dataset, target, device, params['start_point'], \
            params['noise_level'], params['alpha'], params['swap_num'])        
        data_r_init = data_r.clone().detach()

        feature_diff_history = []
        image_diff_history = []
        loss_history = []     
        #######################################################################
        ############### Use PGD to generate noisy images ######################
        for inner in range(params['max_iterations']):
            data_r.requires_grad = True

            feature_diff = loss_feature_diff(model, data, data_r, target, inner, conv_out_layers, \
                params['loss_type'], params['conv'], params['weight_conv'], params['conv_part'], params['feat_weight'])

            image_diff = torch.sum((torch.clamp(data_r, min=0.0, max=1.0) - data)**2) / target.shape[0]
            # image_diff = torch.sum((torch.clamp(data_r, min=0.0, max=1.0) - data_r_init)**2) / target.shape[0]
            loss_fn = feature_diff - params['weight'] * image_diff
            # import ipdb; ipdb.set_trace()
            if params['pgd']==True:
                data_r = pgd_step(loss_fn, data_r, target, params['step_size'])
            else:
                data_r = gd_step(loss_fn, data_r, target, params['step_size'])

            if inner % inner_per_distance_record == 0:
                feature_diff_history.append(feature_diff.item())
                image_diff_history.append(image_diff.item())
                loss_history.append(loss_fn.item())

            if inner % inner_per_image == 0 or inner == max_iterations-1:
                print('Train Epoch:{}\tLoss:{:.6f}\tF_diff:{:.6f}\tI_diff:{:.3f}'.format(inner, loss_fn.item(), 
                    feature_diff.item(), image_diff.item()))

        data_r = torch.clamp(data_r, 0.0, 1.0)
        
        global_feature_diff_history.append(feature_diff_history)
        global_image_diff_history.append(image_diff_history)
        global_loss_history.append(loss_history)

        # save_img(resultpath, data_r, data, idx)
        save_img(resultpath, data_r, data_r_init, idx)
        torch.save(data, resultpath+"images/batch"+str(idx)+"_forg.pt")
        
        for im_idx in range(target.shape[0]):
            target_r.append(target[im_idx])

        # if idx == 0:
        #     break

    target_r = torch.stack(target_r)
    torch.save(target_r, resultpath+"label.pt")

    global_feature_diff_history = torch.tensor(global_feature_diff_history)
    global_image_diff_history = torch.tensor(global_image_diff_history)
    global_loss_history = torch.tensor(global_loss_history)
    
    fig = plt.figure(figsize=(12, 8))
    x_axis = list(range(0, max_iterations-1, inner_per_distance_record))
    plt.plot(global_image_diff_history.mean(dim=0), label = "image_dst")
    plt.plot(global_feature_diff_history.mean(dim=0), label = "feature_dst")
    plt.plot(global_loss_history.mean(dim=0), label = "loss")
    plt.xlabel('x - iterations')
    plt.legend()
    plt.title("dist")
    fig.savefig(resultpath+"fig_loss")
    plt.close()



if __name__=="__main__":
    torch.manual_seed(0)

    datapath="~/dataset"
    modelpath = "./trained_models/best_SVHN_EXTRA_ResNet18None_noise0.0_alpha0.0_results.pt"
    # modelpath = "./trained_models/best_CIFAR100_ResNet18None_noise0.0_alpha0.0_results.pt"
    # modelpath = "./trained_models/last_CIFAR10_VAL_ResNet18_noise0.01_alpha0.0_results.pt"
    # modelpath = './trained_models/last_CIFAR10_VAL_ResNet9_noise0.01_alpha0.0_results.pt'
    resultpath = "./results_test/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default_params = {
        "weight" : 10,
        "weight_conv" : 1.0,
        "noise_level" : 0.2,
        "start_point" : "cifar100",
        # "start_point" : "shift",
        "alpha" : 0.5,
        "step_size" : 0.1,
        "model_type" : "ResNet18SVHNEXTRA",
        "max_iterations": 500,
        "result" : '',
        "defense" : 'None',
        "swap_num" : 128,
        # "conv_part" : 'former',
        "conv_part" : 'latter',
        "loss_type" : "l2", 
        "conv":True, 
        "drop_random":False,
        "pgd":True,
        "feat_weight": 1,
    }

    inner_per_image = 100
    inner_per_distance_record = 10

    batch_size = 128
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])
    # train_loader, _, test_loader = get_cifar10_loader(50, batch_size, transform_train)
    svhn_train = datasets.SVHN('~/dataset', split='train',transform = transform_train, download=True)
    svhn_train = torch.utils.data.Subset(svhn_train, list(range(50000)))
    svhn_extra = datasets.SVHN('~/dataset', split='extra',transform = transform_train, download=True)
    sub_idx = list(range(0, len(svhn_extra), 8))
    valid_dataset = torch.utils.data.Subset(svhn_extra, sub_idx)

    # import ipdb; ipdb.set_trace()
    train_loader = torch.utils.data.DataLoader(svhn_train,batch_size= 128, shuffle=True)
    ########################################################
    ########################################################
    ##############  Load the trained model #################
    ########################################################
    ########################################################
    # model = resnet.ResNet18()
    model = resnet.ResNet18(num_classes=10)
    checkpoint = torch.load(modelpath, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    # test(model, device, test_loader)

    # generate_dataset(model, train_loader, valid_dataset, device, batch_size, resultpath, alpha = alpha, start_point = start_point, 
    #             loss_type=feature_loss, logit = logit_layer, weight = weight, noise_level = noise_level,
    #             max_iterations=max_iterations, inner_per_distance_record=inner_per_distance_record, 
    #             inner_per_image=inner_per_image, batch_stat=batch_stat)
    
    generate_dataset(model, train_loader, valid_dataset, device,batch_size,resultpath, default_params, 
                inner_per_distance_record=inner_per_distance_record, inner_per_image=inner_per_image)
