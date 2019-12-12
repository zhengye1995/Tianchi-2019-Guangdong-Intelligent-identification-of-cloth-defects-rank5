
# 2019广东工业智造创新大赛【赛场一】

队伍：**天池水也太深了**

## 代码github链接：[2019广东工业智造创新大赛 布匹疵点检测 季军解决方案](https://www.datafountain.cn/competitions/366)
## NEW !!!
感谢大家的关注，由于近期很多同学需要数据学习使用，经过和天池的沟通，可以将数据共享给大家学习使用

数据下载地址：

[百度网盘](https://pan.baidu.com/s/1DT8vlFELrjfgczGBZ1yEzQ) (密码：jp7d)

## TIPS:
因为官方给的原始数据压缩包大于4gb，我这里对每一个包进行了分卷压缩，大家注意分卷解压缩使用

## core slides:
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2019-Guangdong-Intelligent-identification-of-cloth-defects-rank5/raw/master/temp_img/%E5%9B%BE%E7%89%871.png)
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2019-Guangdong-Intelligent-identification-of-cloth-defects-rank5/raw/master/temp_img/%E5%9B%BE%E7%89%872.png)
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2019-Guangdong-Intelligent-identification-of-cloth-defects-rank5/raw/master/temp_img/%E5%9B%BE%E7%89%873.png)
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2019-Guangdong-Intelligent-identification-of-cloth-defects-rank5/raw/master/temp_img/%E5%9B%BE%E7%89%874.png)
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2019-Guangdong-Intelligent-identification-of-cloth-defects-rank5/raw/master/temp_img/%E5%9B%BE%E7%89%875.png)
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2019-Guangdong-Intelligent-identification-of-cloth-defects-rank5/raw/master/temp_img/%E5%9B%BE%E7%89%876.png)
## 算法流程&方案介绍 CFRCNN--变化检测思路

+ 输入图片预处理
    - 将代检测图片和模板图片沿通道方向合并, 变为h*w*6的矩阵
    - 归一化
    - resize padding等常规操作
+ 经过CFRCNN模型
    - 基本框架：Cascade-RCNN
    - 输入改变为6个通道的conv, 同时输入待检测图像和对应模板进行变化检测
    - backbone： resnet50
    - cascade 三个head 根据比赛map计算iou进行对应调整
    - 为了解决其中三类面积过大的问题, 额外训练一个小尺度大感受野的专家模型
    - 采用fp16加速训练和增大输入面积来缓解部分小目标问题
+ 后处理
    - NMS
    - 最大score二类后处理, 根据单个图片bbox最高score来对图像进行二次过滤, 判断该图像是否是正常样本

### 创新性

+ 变化检测
    - 根据赛题任务, 采用变化检测思路处理检测任务
    - 调整输入层为样本和模板同时输入 让模型学到目标和模板之间的变化差异, 使模型在切换模板后依然有良好泛化能力
+ 专家模型解决感受野不足问题
    - 为了降低模型复杂度, 没有采用大感受野复杂模型或者增加大anchor, 而是以resnet50为backbone
    - resnet50感受野不足, 并且原始anchor大小不足，导致缝头、缝头印和色差等面积很大的类漏检
    - 训练一个400尺度的大感受野专家模型
        - 去掉面积过小的目标, 保证梯度稳定
+ cascade iou阈值适应赛题map要求
    - cascade 每个head的预测bbox结果在其对应iou阈值的AP上效果最好
    - 根据比赛0.1 0.3 0.5的iou要求, 将cas三个head的iou阈值调整为0.4  0.5  0.6（可能 0.3 0.4 0.5效果更佳，未能尝试）
    - 同时rcnn 正负样例放松overlap要求放松为 0.6 0.2
+ 最大score二类后处理
    - 为了保持map的同时保证acc, 依据单个样本最高score的bbox置信度大小进行二次过滤分出正常图像



## 代码环境及依赖

+ OS: Ubuntu16.10
+ GPU: 2080Ti * 4
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10.0.130
   - cudnn: 7.5.1
   - nvidia driver version: 430.14
+ deeplearning 框架: pytorch1.1.0
+ 其他依赖请参考requirement.txt

## 训练数据准备(后面训练部分会有阐述如何一次性运行，这里只阐述过程)

- **相应文件夹创建准备**

  - 在data目录中创建fabric文件夹
  - 进入fabric文件夹,创建以下文件夹:
  
     annotations
     
     Annotations
     
     defect_Images
     
     template_Images

- **训练数据路径移动**

  - 将 guangdong1_round2_train_part1_20190924,
  
       guangdong1_round2_train_part2_20190924,
  
       guangdong1_round2_train_part3_20190924和
       
       guangdong1_round2_train2_20191004_images中
    
    defect目录中的所有文件夹下的非模板图片复制到 data/fabric/defect_Images 目录下
    
  - 将 guangdong1_round2_train_part1_20190924,
  
       guangdong1_round2_train_part2_20190924,
  
       guangdong1_round2_train_part3_20190924和guangdong1_round2_train2_20191004_images中
    
    defect目录中的所有文件夹复制到 data/fabric/template_Images 目录下
    
    
- **label文件合并及格式转换**

  - 将round2中两个轮次的label文件合并到 anno_train_round2.json中，然后移动到data/fabric/Annotations 目录下
  
  - 将刚才的label文件转换为COCO格式，新的label文件 instances_train_20191004_mmd.json 和 
     instances_train_20191004_mmd_100.json会保存在 data/fabric/annotations 目录下

- **预训练模型下载**
  - 使用mmdetection官方开源的casacde-rcnn-r50-fpn-2x的COCO预训练模型
  - 下载预训练模型后进行转换变为支持CFRCNN模型的预训练模型


## 依赖安装及编译


- **依赖安装编译**

   1. 创建并激活虚拟环境
        conda create -n guangdong python=3.7 -y
        conda activate guangdong

   2. 安装 pytorch
        conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
        
   3. 安装其他依赖
        pip install cython && pip --no-cache-dir install -r requirements.txt
   
   4. 编译cuda op等：
        python setup.py develop
   

## 模型训练及预测
    
   - **训练**
	1. **运行:**
		cd train & ./train.sh

   	2. 训练过程文件及最终权重文件均保存在data目录中

   - **预测**
        1. 线上docker已经提交过预测全部内容，这里依然认为测试数据挂载在/tcdata
        
        2. **运行:**
		./run.sh
   
    

## Contact

    author：rill

    email：18813124313@163.com


