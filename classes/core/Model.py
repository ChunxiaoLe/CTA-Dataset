import os

import torch
from torch import Tensor
import torch.nn as nn
import math

from auxiliary.settings import DEVICE
from classes.losses.AngularLoss import AngularLoss


def arc(target):
  rt = target[0]
  gt = target[1]
  bt = target[2]
  xt = math.acos((rt+gt+bt)/math.sqrt(3*(rt*rt+gt*gt+bt*bt)))/math.sqrt(math.pow(2*rt-gt-bt,2)+3*math.pow(gt-bt, 2))*(2*rt-gt-bt)
  yt = math.acos((rt+gt+bt)/math.sqrt(3*(rt*rt+gt*gt+bt*bt)))/math.sqrt(math.pow(2*rt-gt-bt,2)+3*math.pow(gt-bt, 2))*math.sqrt(3)*(gt-bt)
  return xt, yt

def STD(seq, ns):
    std = 0
    x = []
    y = []
    for i in range(0, ns):
      # xe, ye = processing.arc(estimates[0][i])
      # xt, yt = processing.arc(targets[0][i])
      xe, ye = arc(seq[i])
      x.append(xe)
      y.append(ye)
    xs = sum(x) / len(x)
    ys = sum(y) / len(y)
    x1 = 0
    y1 = 0
    for i in range(0, ns):
      x1 += (x[i] - xs)*(x[i] - xs)/ns
      y1 += (y[i] - ys)*(y[i] - ys)/ns
    std = std + x1 + y1
    std = math.sqrt(std)
    STD = 180 * std / 3.141592653589793
    return STD

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self._device = DEVICE
        self._criterion = AngularLoss(self._device)
        self._optimizer = None
        self._network = None

    def print_network(self):
        print("\n----------------------------------------------------------\n")
        print(self._network)
        print("\n----------------------------------------------------------\n")

    def log_network(self, path_to_log: str):
        open(os.path.join(path_to_log, "network.txt"), 'a+').write(str(self._network))

    def get_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        ae = self._criterion(pred, label)
        # pred2 = []
        # std = 0.0
        # for nbatch in range(0, len(pred1)):
        #     pred2 = []
        #     if nbatch % 3 == 1:
        #         pred2.append(pred1[nbatch-1])
        #         pred2.append(pred1[nbatch])
        #         pred2.append(pred1[nbatch+1])
        #         std += STD(pred2, 3)
        #         #print(pred2, std)
        # std = std / pred.size(0)
        # print(ae, std)
        return ae

    def train_mode(self):
        self._network = self._network.train()

    def evaluation_mode(self):
        self._network = self._network.eval()

    def save(self, path_to_log: str):
        torch.save(self._network.state_dict(), os.path.join(path_to_log, "model.pth"))

    def load(self, path_to_pretrained: str):
        path_to_model = os.path.join(path_to_pretrained, "model.pth")
        self._network.load_state_dict(torch.load(path_to_model, map_location=self._device))

    def set_optimizer(self, learning_rate: float, optimizer_type: str = "adam"):
        optimizers_map = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}
        self._optimizer = optimizers_map[optimizer_type](self._network.parameters(), lr=learning_rate)
