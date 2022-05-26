# The gradient leakage attack, based on the implementation of "Inverting gradients-how easy is it to break privacy in federated learning?" 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import resnet
import lpips
from data_loader import get_cifar10_loader
from torchvision import transforms, datasets
import os
import inversefed
from grad_prune import GradPruneDefense
import argparse

setup = inversefed.utils.system_startup()

def parse_args():
    parser = argparse.ArgumentParser(description='gradattack training')
    parser.add_argument('--cost_fn', default='sim', type=str)
    parser.add_argument('--tv_weight', default=0.0, type=float)
    parser.add_argument('--bn_reg', default=0.0, type=float)
    parser.add_argument('--attack_images', default=20, type=int)
    parser.add_argument('--attack_iter', default=5000, type=int)
    args = parser.parse_args()

    return args

arch = 'ResNet18'

args = parse_args()
cost_fn = args.cost_fn
num_images = 1
attack_images = args.attack_images
batch_size = 128
max_attack_iterations = args.attack_iter
bn_reg = args.bn_reg
tv_weight = args.tv_weight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = torch.nn.CrossEntropyLoss()
percept_loss_fn = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
        percept_loss_fn = percept_loss_fn.cuda()

model = resnet.ResNet18()
model.to(**setup)
model.eval()


dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())


# resultpath = 'resultsResNet18CIFAR100500weight20cifar100max500noise0_2alpha1_0convTruewgt1formerf_wgt0'
resultpath = 'resultsResNet18CIFAR100500weight30cifar100max500noise0_2alpha1_0convTruewgt1wholef_wgt0'
# resultpath = 'resultsResNet18SVHN500weight1cifar100max500noise0_2alpha1_0convTruewgt1latterf_wgt1'
attackpath = 'paper'+cost_fn+'tv'+str(tv_weight)

gen = torch.load(resultpath+'/images/batch0.pt', map_location='cpu').to(device)
org = torch.load(resultpath+'/images/batch0_forg.pt', map_location='cpu').to(setup['device'])
label = torch.load(resultpath+'/label.pt', map_location='cpu').to(setup['device'])

plot_idx = []
offset = 1
# include one image for each class, as representative
for i in range(10):
    plot_idx.append((label[offset:] == i).nonzero()[0].item() + offset)

config = dict(signed=True,
              boxed=True,
              cost_fn=cost_fn,
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=max_attack_iterations,
              total_variation=tv_weight,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss',
              first_bn_multiplier=1,
              bn_reg=bn_reg,
              )

g_loss = []
g_psnr = []
g_lpips = []
g_mse = []
g_fmse = []

attack_outputs = []
ground_truth_list = []

if not os.path.isdir(attackpath):
    os.mkdir(attackpath)

for i in plot_idx:

    ground_truth = org[i].sub(dm).div(ds).unsqueeze(0).contiguous().to(setup['device'])
    labels = label[i].view((-1,)).to(setup['device'])
    
    plot(ground_truth)
    plt.savefig(attackpath+'/attack_org'+str(i)+'.jpg')

    model.zero_grad()

    target_loss = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]

    # Reconstruct
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))

    # Calculate Stat
    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    test_lpips = percept_loss_fn.forward(output.detach(), ground_truth.detach()).squeeze()

    g_loss.append(stats['opt'])
    g_mse.append(test_mse.item())
    g_fmse.append(feat_mse.item())
    g_psnr.append(test_psnr)
    g_lpips.append(test_lpips.item())

    attack_outputs.append(output.detach())
    ground_truth_list.append(ground_truth.detach())

    # Save outputs
    plot(output)
    plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
            f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {test_lpips:2.4f} |")
    plt.savefig(attackpath+'/attack_output'+str(i)+'.jpg')

attack_outputs = torch.stack(attack_outputs)
torch.save(attack_outputs, attackpath+'/attack_output.pt')
ground_truth_list = torch.stack(ground_truth_list)
torch.save(ground_truth_list, attackpath+'/ground_truth.pt')

# import ipdb;ipdb.set_trace()
g_loss = np.array(g_loss)
g_mse = np.array(g_mse)
g_fmse = np.array(g_fmse)
g_psnr = np.array(g_psnr)
g_lpips = np.array(g_lpips)
print('==========================================================')
print(f"Rec.loss: {g_loss.mean():2.4f} +- {g_loss.std():2.4f}")
print(f"MSE: {g_mse.mean():2.4f} +- {g_mse.std():2.4f}")
print(f"F_MSE: {g_fmse.mean():2.4e} +- {g_fmse.std():2.4e}")
print(f"PSNR: {g_psnr.mean():4.2f} +- {g_psnr.std():4.2f}")
print(f"LPIPS: {g_lpips.mean():2.4f} +- {g_lpips.std():2.4f}")
print('==========================================================')
print(f"Rec.loss: {g_loss.min():2.4f}")
print(f"MSE: {g_mse.min():2.4f}")
print(f"F_MSE: {g_fmse.mean():2.4e} +- {g_fmse.std():2.4e}")
print(f"PSNR: {g_psnr.mean():4.2f} +- {g_psnr.std():4.2f}")
print(f"LPIPS: {g_lpips.mean():2.4f} +- {g_lpips.std():2.4f}")
print('==========================================================')
print('Defense: '+cost_fn+'\tBN_Reg: '+str(bn_reg)+'\tTV: '+str(tv_weight))
