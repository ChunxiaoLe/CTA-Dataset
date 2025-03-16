import json
import os
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

#from auxiliary.settings import get_device


def log_experiment(model_type: str, data_folder: str, lr: float, path_to_log: str):
    experiment_data = {
        "model_type": model_type,
        "data_folder": data_folder,
        "learning_rate": lr,
        "train_time": 0,
        "val_time": 0
    }
    json.dump(experiment_data, open(path_to_log, 'w'), indent=2)


def log_time(time: float, time_type: str, path_to_log: str):
    data = json.load(open(path_to_log, 'r'))
    data["{}_time".format(time_type)] += time
    open(path_to_log, 'w+').write(json.dumps(data, indent=2))


def log_metrics(train_loss: float, val_loss: float, current_metrics: Dict, best_metrics: Dict, path_to_log: str):
    log_data = pd.DataFrame({
        "train_loss": [train_loss],
        "val_loss": [val_loss],
        "best_mean": best_metrics["mean"],
        "best_median": best_metrics["median"],
        "best_trimean": best_metrics["trimean"],
        "best_bst25": best_metrics["bst25"],
        "best_wst25": best_metrics["wst25"],
        "best_wst5": best_metrics["wst5"],
        **{k: [v] for k, v in current_metrics.items()}
    })
    log_data.to_csv(path_to_log,
                    mode='a',
                    header=log_data.keys() if not os.path.exists(path_to_log) else False,
                    index=False)


def print_val_metrics(current_metrics: Dict, best_metrics: Dict):
    print(" Mean ......... : {:.4f} (Best: {:.4f})".format(current_metrics["mean"], best_metrics["mean"]))
    print(" Median ....... : {:.4f} (Best: {:.4f})".format(current_metrics["median"], best_metrics["median"]))
    print(" Trimean ...... : {:.4f} (Best: {:.4f})".format(current_metrics["trimean"], best_metrics["trimean"]))
    print(" Best 25% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["bst25"], best_metrics["bst25"]))
    print(" Worst 25% .... : {:.4f} (Best: {:.4f})".format(current_metrics["wst25"], best_metrics["wst25"]))
    print(" Worst 5% ..... : {:.4f} (Best: {:.4f})".format(current_metrics["wst5"], best_metrics["wst5"]))


def print_test_metrics(metrics: Union[Dict, Tuple]):
    if isinstance(metrics, Dict):
        print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
        print(" Median .......... : {:.4f}".format(metrics["median"]))
        print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
        print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
        print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
        print(" Worst 5% ........ : {:.4f} \n".format(metrics["wst5"]))
    else:
        metrics1, metrics2, metrics3 = metrics
        print("\n Mean ............ : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
              .format(metrics1["mean"], metrics2["mean"], metrics3["mean"]))
        print(" Median .......... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
              .format(metrics1["median"], metrics2["median"], metrics3["median"]))
        print(" Trimean ......... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
              .format(metrics1["trimean"], metrics2["trimean"], metrics3["trimean"]))
        print(" Best 25% ........ : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
              .format(metrics1["bst25"], metrics2["bst25"], metrics3["bst25"]))
        print(" Worst 25% ....... : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ]"
              .format(metrics1["wst25"], metrics2["wst25"], metrics3["wst25"]))
        print(" Worst 5% ........ : [ s1: {:.4f} | s2: {:.4f} | s3: {:.4f} ] \n"
              .format(metrics1["wst5"], metrics2["wst5"], metrics3["wst5"]))


def correct(img: np.ndarray, illuminant: torch.Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = TF.to_tensor(img)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(torch.Tensor([3])).to(get_device())
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    linear_image = torch.pow(normalized_img, 1.0 / 2.2)
    return TF.to_pil_image(linear_image.squeeze(), mode="RGB")


def linear_to_nonlinear(img: Image) -> Image:
    return TF.to_pil_image(torch.pow(TF.to_tensor(img), 1.0 / 2.2).squeeze(), mode="RGB")


def rgb_to_bgr(color: np.ndarray) -> np.ndarray:
    return color[::-1]


def brg_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 4:
        return img[:, :, :, ::-1]
    elif len(img.shape) == 3:
        return img[:, :, ::-1]
    raise ValueError("Bad image shape detected in BRG to RGB conversion: {}".format(img.shape))


def hwc_chw(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 4:
        return img.transpose(0, 3, 1, 2)
    elif len(img.shape) == 3:
        return img.transpose(2, 0, 1)
    raise ValueError("Bad image shape detected in HWC to CHW conversion: {}".format(img.shape))


def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return np.power(img, (1.0 / gamma))
