## Pruning implementation on top of official yolov7.

Run pruning:
```
!python prune.py --prune_ratio 0.5 --steps 2 --data Pascal-Voc-2007-1/data.yaml --cfg /cfg/training/yolov7.yaml --device 0 --weights 'yolov7.pt' --batch-size 32 --epochs 3 --cache-images --name exp_voc --img-size 416

```