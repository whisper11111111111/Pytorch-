# PyTorch手写数字识别系统
这是一个基于PyTorch实现的MNIST手写数字识别系统，并包含了图形用户界面应用程序。

（AI辅助生成，可以用来水《机器视觉》课程大作业）
# 安装
## 创建虚拟环境（建议）
``
conda create -n handwriting_env``
``conda actevate handwriting_env``
## 模型训练
``python mnist.py --mode train``  

(如果数据集下载失败，可自行从kaggle上下载，并注意修改数据集路径)

``
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
``  

``
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
``
## 使用GUI应用程序
``python mnist_gui.py``
