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

## Res-net
![res-net-1](img/resnet.png)
![res-net-2](img/resnet_structure.png)
## Inception-net
![inception-net](img/inceptionnet.png)
### V1
![inception-net-v1](img/inceptionnet_v1_structure.png)
### V2
![inception-net-v2](img/inceptionnet_v2_structure.png)
### V3
![inception-net-v3](img/inceptionnet_v3_structure.png)
### V4
![inception-net-v4](img/inceptionnet_v4_structure.png)
## Mobile-net
![mobile-net-1](img/mobilenet_structure_1.png)
![mobile-net-2](img/mobilenet_structure_2.png)
## cnn tricks


