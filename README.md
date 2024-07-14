## Pruning implementation on top of official yolov7.

Run pruning:

```
!python pruner.py --prune_ratio 0.5 --steps 2 --data Pascal-Voc-2007-1/data.yaml --cfg ./cfg/training/yolov7.yaml --device 0 --weights /kaggle/working/best.pt --batch-size 32 --epochs 1 --cache-images --name exp_voc --img-size 416

```

Prune ratio defines percentage of model channels to prune.
Steps define iterative steps to prune the model.