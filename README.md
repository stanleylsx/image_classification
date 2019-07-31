# Image classification
This project contains various CNN for image classification.
## Dataset
cifar-10
## Alex-net
![alex-net](img/alexnet_structure.jpg)
论文：《ImageNet Classification with Deep Convolutional Neural Networks》   
意义：相比传统结构有了更高的分类准确度，引爆深度学习，首次使用Relu。   
结构：2-GPU并行结构；1、2、5卷积层后跟随max-pooling层；两个全连接使用dropout；总共8层神经网络。
## Vgg-net
论文：《Very Deep Convolutional Networks for Large-Scale Image Recognition》    
意义：分类问题第二，物体检测第一(ImageNet2014)。  
结构：更深的网络结构；使用3x3的卷积核和1x1的卷积核；每经过一个pooling层，通道数目翻倍。
两个3x3卷积层视野率等于一个5x5卷积核，多一次线性变换且参数数量降低28%，1x1的卷积核可以看做在对应通道上的非线性变换，有通道降维的作用。  
![vgg-net](img/vggnet_structure.jpg)  
本项目实现的代码基于Cifar数据集，实现了简单层次的vggnet，两次卷积加一个池化层并重复三次并且最后全连接层只实现了一层。
## Res-net 
![res-net-1](img/resnet_1.png) 
论文：《Deep Residual Learning for Image Recognition》   
意义：ILSVRC2015分类比赛冠军，解决深层次网络训练问题。  
结构：加入恒等变换子结构，identity部分是恒等变换，F(x)为残差学习，学习使得F(x)趋向0，从而忽略深度。  
不同的ResNet有不同结构，如图ResNet-34和ResNet-101是两种常用结构。
![res-net-2](img/resnet.png)   
所有的网络结构可以通用描述为：
* 先用一个步长为2的卷积层。
* 经过一个3x3的max_pooling层
* 经过残差结构
* 没有中间的全连接层，直接到输出。

![res-net-3](img/resnet_structure.png)
上图表格中有更多的结构，从各个结构可以看出ResNet强化了卷积层，弱化了全连接层，维持了参数平衡。  
特点：残差结构使得网络需要的学习的知识变少，容易学习；残差结构使得每一层的数据分布接近，容易学习。

代码同样使用Cifer数据集，由于数据集图片较小，所以在输入到残差结构前的卷积层步长设定为1，且没有经过3x3的max_pooling层。
每个残差块都由两个3x3卷积核组成，总共有三个残差层，残差块的个数分别为2、3、2。

## Inception-net
论文：《Rethinking the Inception Architecture for Computer Vision》    
意义：主要是工程的优化，使得同样的参数数量训练更加的效率。一方面解决更深的网络过拟合，另外一方面解决更深的网络有更大计算量的问题。
结构：主要是v1~v4四个结构。
![inception-net](img/inceptionnet.png)
### V1
![inception-net-v1](img/inceptionnet_v1_structure.png)

![inception-net-v1_1](img/inceptionnet_v1_structure_1.png)
采用分组卷积，组与组之间的数据在分组计算时候不会交叉。一层上同时使用多种卷积核，看到各层的feature；不同组之间的feature不交叉计算，减少了计算量。
### V2
![inception-net-v2](img/inceptionnet_v2_structure.png)
引入3x3的卷积核做同等卷积替换，两个3x3卷积核的视野域和一个5x5的相同。
### V3
![inception-net-v3](img/inceptionnet_v3_structure.png)
进一步的做同等卷积替换，一个3x3的卷积核的视野域等同于一个1x3的卷积核加上一个3x1卷积核。
### V4
![inception-net-v4](img/inceptionnet_v4_structure.png)
使用和ResNet同样的思想，引入skip connection，可解决深层次网络训练问题。

项目中基于v1结构实现了简单的InceptionNet，受限于数据集图片的大小，层次不深，各层的步长和核大小被调整。
结构为conv1(3x3/1)->max_pooling1(2x2/2)->inception_2a->inception_2b->max_pooling2(2x2/2)->inception_3a->inception_3a->max_pooling2(2x2/2)->dense，
每个inception都会增加通道的数目，但是图片的大小维持不变。
## Mobile-net
论文：《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》
意义：引入深度可分离的卷积，进一步降低参数。
![mobile-net-1](img/mobilenet_structure_1.png)
下图设定输入输出通道数为300，对于一个普通的3x3卷积，它需要3*3*300*300个参数。
若分组卷积，对分组卷积来看它的参数为3*3*100*100*3。参数降低1/3。
![mobile-net-2](img/mobilenet_structure_2.png)

![mobile-net-3](img/mobilenet_structure_3.png)
MobileNet将分组卷积做到极致，如上图所示，每一个3x3卷积核只管一个通道。
实现和InceptionNet结构差不多，要注意的是在通过深度可分离卷积块的时候(separable_X)，将通道分割开分别送入一个3x3的卷积核，再把它们的输出拼接起来。
## Cnn tricks
### Activation
### Initializer
### Optimizer
### Data augmentation
### Fine tune


