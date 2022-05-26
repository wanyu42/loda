import torch
import trainmodel
from torchvision import transforms
import argparse
from data_loader import *
from dataset_gen import *


# torch.manual_seed(0)
torch.manual_seed(123)

def parse_args():
    parser = argparse.ArgumentParser(description='Defense against Deep Leakage.')
    parser.add_argument('--weight', type=float, default="1e-2",
                    help='the weight for image space difference.')
    parser.add_argument('--feat_weight', type=float, default="1",
                    help='the weight for feature matching.')
    parser.add_argument('--noise', type=float,default="0.2",
                    help='the standard deviation of initial gaussian noise.')
    parser.add_argument('--loss', type=str, default="l2",
                    help='The feature distance metric. support type: l2, kl')  
    parser.add_argument('--starting', type=str, default="mixup",
                    help='The starting point. support type: gaussian, inclass, outclass')
    parser.add_argument('--alpha', type=float, default="1.0",
                    help='The weight for the mixup, with 0.0 indicates original dataset')                       
    parser.add_argument('--result', type=str, default="",
                    help='The folder name suffix for results')
    parser.add_argument('--max_iter', type=int, default="1000",
                    help='The max iterations for data optimization')
    parser.add_argument('--step_size', type=float, default="0.1",
                    help='The step size for dataset generation')
    parser.add_argument('--pgd', type=str, default="True",
                    help='True: The optimization for dataset generation is pgd, False: standard gd')
    parser.add_argument('--conv', type=str, default="False",
                    help='True: add conv matching to objective for dataset generation')
    parser.add_argument('--weight_conv', type=float, default="0.0",
                    help='The weight for conv matching. Default to 0.0')
    parser.add_argument('--conv_part', type=str, default="former",
                    help='former means the first 5 layers of conv\tmiddle means middle 5 layers\tlatter means last 5 layers')
    parser.add_argument('--model_type', type=str, default="ResNet18",
                    help='ResNet18: The model used for dataset generation')
    parser.add_argument('--random', type=str, default="False",
                    help='True with dropout, False without drop in generation process. default False')
    parser.add_argument('--defense', type=str, default="None",
                    help='None: no defense\nGradPrune_099: Prune 0.99 gradients')
    parser.add_argument('--swap_num', type=int, default="128",
                    help='Used only in starting=half. Default 128: equivalent to reproduce')    
    args = parser.parse_args()

    params = {  
            "weight" : args.weight,
            "noise_level" : args.noise,
            "start_point" : args.starting,
            "alpha" : args.alpha,
            "step_size" : args.step_size,
            "model_type" : args.model_type,
            "max_iterations": args.max_iter,
            "result" : args.result,
            "defense" : args.defense,
            "swap_num" : args.swap_num,
            "weight_conv" : args.weight_conv,
            "conv_part" : args.conv_part,
            "feat_weight" : args.feat_weight,
            }

    if args.random == "True":
        params['dropout_random'] = True
    else:
        params['dropout_random'] = False
    if args.pgd == "True":
        params['pgd'] = True
    else:
        params['pgd'] = False
    if args.conv == "True":
        params['conv'] = True
    else:
        params['conv'] = False
    
    if args.loss == "l2":
        params['loss_type'] = "l2"
    else:
        params['loss_type'] = "kl"
        
    return args, params


args, params = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
originalDataPath="~/dataset"

if params['start_point'] == "half":
    resultpath = "./results"+params['model_type']+"swap"+str(params['swap_num'])+params['result'] +"/"
else:
    resultpath = "./results"+params['model_type'] + params['result'] +"/"
########################################################
##############  Load the trained model #################
########################################################
if params['model_type'] == 'ResNet18':   
    modelpath = "./trained_models/CIFAR10_VAL_ResNet18_epoch_200.pt"
    model = resnet.ResNet18(random=params['dropout_random'])
elif params['model_type'] == 'ResNet9':
    modelpath = './trained_models/last_CIFAR10_VAL_ResNet9_noise0.01_alpha0.0_results.pt'
    model = resnet.ResNet9(random=params['dropout_random'])
elif params['model_type'] == 'ResNet50':
    modelpath = './trained_models/last_CIFAR10_VAL_ResNet50_noise0.0_alpha0.0_results.pt'
    model = resnet.ResNet50(random=params['dropout_random'])
elif params['model_type'] == 'ResNet18CIFAR100':
    modelpath = './trained_models/best_CIFAR100_ResNet18None_noise0.0_alpha0.0_results.pt'
    model = resnet.ResNet18(num_classes=100, random=params['dropout_random'])
checkpoint = torch.load(modelpath, map_location=device)
model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("================================================================")
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
print("================================================================")

########################################################################
########################################################################
####################### Generate Dataset ###############################
########################################################################
########################################################################
inner_per_image = 100
inner_per_distance_record = 10
batch_size = 128

transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])
train_loader, _, test_loader = get_cifar10_loader(50, batch_size, transform_train)
transform_valid = transforms.Compose([transforms.ToTensor(),])
valid_dataset = CIFAR10(50, 0.0, mode = "valid", transform=transform_valid)


if not os.path.exists(resultpath + "label.pt"):
        generate_dataset(model, train_loader, valid_dataset, device,batch_size,resultpath, params, 
                inner_per_distance_record=inner_per_distance_record, inner_per_image=inner_per_image)
else:
        print("Already exist dataset")

########################################################################
########################################################################
####################### Train new model ################################
########################################################################
########################################################################
print("================================================================")
print("================================================================")
print("================================================================")

transform_new = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

trainmodel.train(params['model_type'], 'CIFAR10_MASKED', device, 200, data_path = resultpath, 
        transform = transform_new, write_accuracy=True, args = args, 
        save_model=False, defense=params['defense'])


print("================================================================")
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
print("================================================================")
print('Defense: '+params['defense'])

