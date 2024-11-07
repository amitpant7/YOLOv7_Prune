import argparse
import time
import torch
from numpy import random
from pathlib import Path

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel


def calculate_fps(weights, img_size=416, device="cpu", trace=True):
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Run inference on a random image
    img = torch.zeros((1, 3, img_size, img_size)).to(device)
    if half:
        img = img.half()

    # Warmup
    model(img)  # run once

    # Measure FPS
    t0 = time.time()
    for _ in range(100):  # run 100 iterations
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        t2 = time_synchronized()

    t_total = time.time() - t0
    fps = 100 / t_total
    print(f"FPS: {fps:.2f}")

    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov7.pt", help="model.pt path"
    )
    parser.add_argument(
        "--img-size", type=int, default=416, help="inference size (pixels)"
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    opt = parser.parse_args()

    calculate_fps(
        opt.weights, img_size=opt.img_size, device=opt.device, trace=not opt.no_trace
    )
