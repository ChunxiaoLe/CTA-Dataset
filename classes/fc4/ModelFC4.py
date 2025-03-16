import os
from typing import Union, Tuple

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import transforms
import torch.nn as nn
import torch

from auxiliary.settings import USE_CONFIDENCE_WEIGHTED_POOLING
from auxiliary.utils import correct, rescale, scale
from classes.core.Model import Model
from classes.fc4.FC4 import FC4





class ModelFC4(Model):

    def __init__(self):
        super(ModelFC4, self).__init__()
        self._network = FC4().to(self._device)


    def predict_hook(self, img: Tensor, mimic:Tensor=None, return_steps: bool = False) -> Union[Tensor, Tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, rgb, confidence = self._network(img, mimic)
            if return_steps:
                return pred, rgb, confidence
            return pred
        else:
            attn_values = []

            def hook_fn(module, input, output):
                # if hasattr(module, 'attn'):
                #     # 将 sim 值添加到列表中
                #     attn_values.append(module.attn.clone().detach())
                if hasattr(module, 'x'):
                    # 将 sim 值添加到列表中
                    attn_values.append(module.x.clone().detach())

            # 获取最后一个 RepViTBlock 【-1】
            last_block = self._network.backbone1.features[-1]

            # 从最后一个 RepViTBlock 中获取 CrossAttention 层
            cross_attention_layer = last_block.crossattention

            # 在 CrossAttention 层上注册 forward hook
            hook = cross_attention_layer.register_forward_hook(hook_fn)

            pred = self._network(img, mimic)
        return pred, attn_values

    def predict(self, img: Tensor, mimic:Tensor=None, return_steps: bool = False) -> Union[Tensor, Tuple]:
        """
        Performs inference on the input image using the FC4 method.
        @param img: the image for which an illuminant colour has to be estimated
        @param return_steps: whether or not to also return the per-patch estimates and confidence weights. When this
        flag is set to True, confidence-weighted pooling must be active)
        @return: the colour estimate as a Tensor. If "return_steps" is set to true, the per-path colour estimates and
        the confidence weights are also returned (used for visualizations)
        """
        if USE_CONFIDENCE_WEIGHTED_POOLING:
            pred, rgb, confidence = self._network(img)
            if return_steps:
                return pred, rgb, confidence
            return pred
        else:
            pred = self._network(img, mimic)

        return pred

    def optimize(self, img: Tensor, mimic: Tensor, label: Tensor) -> float:
        self._optimizer.zero_grad()
        pred = self.predict(img, mimic)

        loss = self.get_loss(pred, label)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def save_vis(self, model_output: dict, path_to_plot: str):
        model_output = {k: v.clone().detach().to(self._device) for k, v in model_output.items()}

        img, label, pred = model_output["img"], model_output["label"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(img.squeeze()).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]
        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)
        #一个通过置信度加权的原始图像结果
        masked_original = scale(F.to_tensor(original).to(self._device).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_loss(pred, label)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')
