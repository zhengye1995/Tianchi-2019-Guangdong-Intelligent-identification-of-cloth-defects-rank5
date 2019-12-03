python inference/generate_test_json_round2.py
./inference/dist_test.sh config/fabric_defect/cascade_rcnn_r50_fpn_70e.py weights/cascade_rcnn_r50_fpn_70e/latest.pth 1 --json_out data/result_map.json
wait
./inference/dist_test.sh config/fabric_defect/cascade_rcnn_r50_fpn_400.py weights/cascade_rcnn_r50_fpn_400/latest.pth 1 --json_out data/result_zj.json
python inference/json2submit1.py
python inference/json2submit3.py
wait
python inference/merger_json2.py
