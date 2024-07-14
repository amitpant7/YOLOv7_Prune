## use vitis ai copy left zooo 

## commands, use voc yamls

cd yolov7/
# run calibration & test & dump xmodel

python test_nndct.py --data data/voc.yaml --img 416 --batch 1 --conf 0.001 --iou 0.65 --device 'cpu' --weights model.pt --name yolov7_640_val --quant_mode calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish

python test_nndct.py --data data/voc.yaml --img 416 --batch 1 --conf 0.001 --iou 0.65 --device 'cpu' --weights model.pt --name yolov7_640_val --quant_mode test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish