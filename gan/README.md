# GAN、WGAN、WGAN-GP实现

### 训练

    bash run.sh # 将数据集放在根目录下
    
或

    python main.py --cuda --dataroot . 
   
   
其中可选参数如下

- **dataroot** 数据根目录，默认为.
- **batchSize** batch大小，默认为400
- **isize** 输入大小，默认为2
- **nz** 生成的随机噪声大小，默认为10
- **ngf** 生成器中间节点个数，默认为1000
- **ndf** 判别器中间节点个数，默认为1000
- **niter** 迭代次数，默认为200
- **lrD** 判别器学习率，默认为0.00018
- **lrG** 生成器学习率，默认为0.00018
- **b1** Adam优化器的beta1值，默认为0.5，仅针对优化器为Adam的情况
- **clamp_lower** 权重裁剪的下界，默认为-0.01，仅针对模型为wgan的情况
- **clamp_upper** 权重裁剪的上界，默认为0.01，仅针对模型为wgan的情况
- **mode** 选择的模型，gan/wgan/wgan-gp，默认为gan
- **experiment** 图片存放目录
- **optim** 选择的优化器，adam/rmsprop，默认为rmsprop

### 使用tensorboard查看训练结果

在根目录下执行

    tensorboard --logdir runs

根据提示在浏览器打开对应网页即可查看[若无法打开网页可尝试检查代理是否关闭]
