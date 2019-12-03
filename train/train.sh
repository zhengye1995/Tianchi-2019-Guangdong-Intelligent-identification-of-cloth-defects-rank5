#!/bin/bash
mkdir -p ../data/fabric/defect_Images
mkdir -p ../data/fabric/template_Images
mkdir -p ../data/fabric/annotations
mkdir -p ../data/fabric/Annotations
mkdir -p ../data/pretrained
cp ../data/guangdong1_round2_train_part1_20190924/defect/*/*.jpg ../data/fabric/defect_Images/ & cp -r ../data/guangdong1_round2_train_part2_20190924/defect/*/*.jpg ../data/fabric/defect_Images/ & cp ../data/guangdong1_round2_train_part3_20190924/defect/*/*.jpg ../data/fabric/defect_Images/ & cp ../data/guangdong1_round2_train2_20191004_images/defect/*/*.jpg ../data/fabric/defect_Images/
wait
rm ../data/fabric/defect_Images/template*
cp -r ../data/guangdong1_round2_train_part1_20190924/defect/* ../data/fabric/template_Images/ & cp -r ../data/guangdong1_round2_train_part2_20190924/defect/* ../data/fabric/template_Images/ & cp -r ../data/guangdong1_round2_train_part3_20190924/defect/* ../data/fabric/template_Images/ & cp -r ../data/guangdong1_round2_train2_20191004_images/defect/* ../data/fabric/template_Images/
wait
cp ../data/guangdong1_round2_train_part1_20190924/Annotations/anno_train.json ../data/fabric/Annotations/anno_train_20190925.json
cp ../data/guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json ../data/fabric/Annotations/anno_train_20191008.json
python merage_data_json.py
python guangdong_round2.py
python guangdong_round2_100.py
wget https://open-mmlab.oss-cn-beijing.aliyuncs.com/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth -O ../data/pretrained/
python transorfarm_concatenate_model.py
CUDA_VISIBLE_DEVICES=0,1,2,3 ./dist_train.sh ../config/fabric_defect/cascade_rcnn_r50_fpn_70e.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./dist_train.sh ../config/fabric_defect/cascade_rcnn_r50_fpn_400.py 4
python publish_model.py ../data/work_dirs/cascade_rcnn_r50_fpn_70e/latest.pth ../data/work_dirs/cascade_rcnn_r50_fpn_70e/latest-submit.pth
python publish_model.py ../data/work_dirs/cascade_rcnn_r50_fpn_400/latest.pth ../data/work_dirs/cascade_rcnn_r50_fpn_400/latest-submit.pth




