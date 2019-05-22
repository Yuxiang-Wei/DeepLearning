# 简单的AlexNet实现

### 数据集

使用torchvision提供的CIFAR10数据集，存放在根目录下data目录中，结构如下:

    data
     └── cifar-10-python.tar.gz
        
或

    data
     └── cifar-10-batches-py

### 训练

    python alexnet.py
    
或者可以指定训练使用的参数

    python alexnet.py --epochs 10 --batch_size 50 --lr 1e-3
    
训练好的模型存放在./checkpoints下，若要在训练开始前加载已训练好的模型，可以使用--load 参数指定

    python alexnet.py --load True
    
### 使用tensorboard查看训练结果

在根目录下执行

    tensorboard --logdir runs

根据提示在浏览器打开对应网页即可查看[若无法打开网页可尝试检查代理是否关闭]
