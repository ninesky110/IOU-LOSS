# IOU-LOSS
针对语义分割二分类问题，可用IOU损失替代交叉熵损失
该程序依据的论文为：Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation
该论文的思想对应的程序为注释段的内容，即只计算正样本的IOU损失；
程序中非注释段采用的是分别计算正样本和负样本的IOU，然后求平均；
