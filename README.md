# MSGSA
Multi-Stage Network with Geometric Semantic Attention for Two-View Correspondence Learning

## MSGSA implementation
Pytorch implementation of MSGSA. 
We provided two versions of the code: one using the weighted eight-point algorithm similar to OANet, and the other using the improved weighted eight-point algorithm similar to SGA-Net.

## Requirements
Please use Python 3.8, opencv-contrib-python (4.6.0.66) and Pytorch (>= 1.10.0). Other dependencies should be easily installed through pip or conda.

## Datasets
If you need YFCC100M and SUN3D datasets, You can visit the code at https://github.com/zjhthu/OANet.git.

## Test pretrained model
We provide the model trained on YFCC100M and SUN3D described in our paper. You can edit the 'config.py' file to specify the path to your datasets. Afterwards, you can run the test script to obtain results.
```shell
python main.py --run_mode=test --model_path=./weights/yfcc100m/model_best.pth --res_path=./results/yfcc/test
```
## Train
To initiate training, please modify the 'config.py' file to specify the path. Subsequently, execute the train script to commence training. You can also specify the path in the training script below, but it is recommended to directly modify the path configuration in the 'config.py' file.
```shell
python main.py --run_mode=train 
```

## Citation
```shell
@article{lin2024multistage,
  title = {Multi-stage network with geometric semantic attention for two-view correspondence learning},
  author = {Lin, Shuyuan and Chen, Xiao and Xiao, Guobao and Wang, Hanzi and Huang, Feiran and Weng, Jian},
  journal = {IEEE Transactions on Image Processing},
  volume = {33},
  pages = {3031--3046},
  year = {2024},
}
```