# MLP

使用pytorch实现的简单MLP，数据集为MNIST数据集，模型结构如下：

    MLP(
      (linear1): Linear(in_features=784, out_features=512, bias=True)
      (linear2): Linear(in_features=512, out_features=256, bias=True)
      (linear3): Linear(in_features=256, out_features=10, bias=True)
    )


### 训练模型

    python mlp.py --train True --epochs 10 --batch_size 50 --lr 1e-3
    
其中后3个参数为可选
 
### 测试模型
 
    python mlp.py --test True --batch_size 50 
   