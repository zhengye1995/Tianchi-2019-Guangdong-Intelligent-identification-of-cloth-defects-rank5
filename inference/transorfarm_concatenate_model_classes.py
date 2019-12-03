# for cascade rcnn
import torch
import os
from torch.nn import init
import numpy as np
num_classes = 16
model_name = "data/pretrained/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth"

model_coco = torch.load(model_name)
# print(model_coco["state_dict"]["backbone.conv1.weight"].shape)
# print(model_coco["state_dict"]["backbone.bn1.weight"].shape)
# print(model_coco["state_dict"]["backbone.bn1.bias"].shape)
# print(model_coco["state_dict"]["backbone.bn1.running_mean"].shape)
# print(model_coco["state_dict"]["backbone.bn1.running_var"].shape)
# print(model_coco["state_dict"]["backbone.bn1.num_batches_tracked"].shape)
# input()

# bbox_weight_example = model_coco["state_dict"]["bbox_head.fc_cls.weight"]
# bbox_bias_example = model_coco["state_dict"]["bbox_head.fc_cls.bias"]
# model_coco["state_dict"]["bbox_head.fc_cls.weight"] = init.normal_(torch.empty(bbox_weight_example[:num_classes, :].shape))
# model_coco["state_dict"]["bbox_head.fc_cls.bias"] = init.normal_(torch.empty(bbox_bias_example[:num_classes].shape))
# save new model
# weight
model_coco["state_dict"]["backbone.conv1.weight"] = torch.cat([model_coco["state_dict"]["backbone.conv1.weight"]]*2, axis=1)
print(model_coco["state_dict"]["backbone.conv1.weight"].shape)
# model_coco["state_dict"]["bbox_head.fc_cls.weight"].resize_(num_classes,1024)
# model_coco["state_dict"]["bbox_head.fc_reg.weight"].resize_(num_classes*4,1024)

# bias
# model_coco["state_dict"]["bbox_head.fc_cls.bias"].resize_(num_classes)
# model_coco["state_dict"]["bbox_head.fc_reg.bias"].resize_(num_classes*4)
#save new model
torch.save(model_coco,"data/pretrained/concatenate_coco_pretrained_"+os.path.basename(model_name).split('.')[0]+".pth")


# model_name = "data/pretrained/retinanet_x101_32x4d_fpn_2x_20181218-8596452d.pth"
# model_coco = torch.load(model_name)
#
# # weight
# model_coco["state_dict"]["bbox_head.retina_cls.weight"].resize_((num_classes-1)*9, 256, 3, 3)
# model_coco["state_dict"]["bbox_head.retina_reg.weight"].resize_(36, 256, 3, 3)
#
# # bias
# model_coco["state_dict"]["bbox_head.retina_cls.bias"].resize_((num_classes-1)*9)
# model_coco["state_dict"]["bbox_head.retina_reg.bias"].resize_(36)
#save new model
# torch.save(model_coco,"data/pretrained/coco_pretrained_"+os.path.basename(model_name).split('.')[0]+"_weights_classes_%d.pth"%num_classes)

