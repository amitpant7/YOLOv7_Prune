## Pruning implementation on top of official yolov7.

Run pruning:

```
!python pruner.py --prune_ratio 0.5 --steps 2 --data Pascal-Voc-2007-1/data.yaml --cfg ./cfg/training/yolov7.yaml --device 0 --weights /kaggle/working/best.pt --batch-size 32 --epochs 1 --cache-images --name exp_voc --img-size 416

```

Prune ratio defines percentage of model channels to prune.
Steps define iterative steps to prune the model.

The iterative pruning (using pruner.py) is not performing due to some issues in prorper retraining instead pruner_one_shot can be used to prune the model and, retrain it for few epochs. 

The pruned models yaml need to be updated using `generate_prune_yaml.py`, this yaml can be used for further retraining the pruned model or pruning it to higher ratio achieving iterative pruning.