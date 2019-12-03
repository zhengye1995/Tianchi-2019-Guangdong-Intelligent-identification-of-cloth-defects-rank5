# for cascade rcnn
import torch
import os
from torch.nn import init
import numpy as np

model_name = "../data/pretrained/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth"

model_coco = torch.load(model_name)

# weight
model_coco["state_dict"]["backbone.conv1.weight"] = torch.cat([model_coco["state_dict"]["backbone.conv1.weight"]]*2, dim=1)
print(model_coco["state_dict"]["backbone.conv1.weight"].shape)

#save new model
torch.save(model_coco,"../data/pretrained/concatenate_coco_pretrained_"+os.path.basename(model_name).split('.')[0]+".pth")

