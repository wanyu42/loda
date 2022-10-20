# loda
This is a repository for implementation of LODA and proposed adaptive attack.

To reproduce the generation of obscured data via **LODA1**, run the following command:
```
python main_cifar.py --seed $SEED --weight 20 --noise 0.2 --starting cifar100 --alpha 1.0 --conv True --weight_conv 1 --conv_part former --max_iter 500 --defense None --result $RESULT --model_type ResNet18CIFAR100 --feat_weight 0
```
To reproduce the generation of obscured data via **LODA2**, run the following command:
```
python main_cifar.py --seed $SEED --weight 30 --noise 0.2 --starting cifar100 --alpha 1.0 --conv True --weight_conv 1 --conv_part whole --max_iter 500 --defense None --result $RESULT --model_type ResNet18CIFAR100 --feat_weight 0
```
We also examine the explanability of the model trained on LODA obscured data. The below images are explanation results on clean test images. 

Explanability Results\
![](./Original_Images_random_100.png?raw=true "Title")
![](./SmoothGradCAM++_Standard_model_random_100.png?raw=true "Title")
![](./SmoothGradCAM++_LODA_model_random_100.png?raw=true "Title")
