# MSGSA
Multi-Stage Network with Geometric Semantic Attention for Two-View Correspondence Learning
# OANet implementation
Pytorch implementation of MSGSA
# Requirements
Please use Python 3.8, opencv-contrib-python (4.6.0.66) and Pytorch (>= 1.10.0). Other dependencies should be easily installed through pip or conda.
# Datasets
If you need YFCC100M and SUN3D datasets, You can visit the code at https://github.com/zjhthu/OANet.git.

# Test pretrained model
We provide the model trained on YFCC100M and SUN3D described in our paper. You can edit the 'config.py' file to specify the path to your datasets. Afterwards, you can run the test script to obtain results. 
```shell
python main.py --run_mode=test --model_path=./weights/yfcc100m/model_best.pth --res_path=./results/yfcc/test
```
# Train
To initiate training, please modify the 'config.py' file to specify the path. Subsequently, execute the train script to commence training. You can also specify the path in the training script below, but it is recommended to directly modify the path configuration in the 'config.py' file.
```shell
python main.py --run_mode=train 
```
# Acknowledgement
This code is heavily borrowed from OANet. If you use the part of code related to data generation, testing and evaluation, you should cite these paper and follow their license.
```shell
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={5845--5854},
  year={2019}
}
```