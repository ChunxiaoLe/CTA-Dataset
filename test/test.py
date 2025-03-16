import argparse
import os
from time import time, perf_counter
import sys
import math
import re
from torch import Tensor
import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader

sys.path.append('./')
from auxiliary.settings import DEVICE
from classes.core.LossTracker import LossTracker
from classes.core.Evaluator import Evaluator
from classes.data.OurDataset import CtaDataset
from classes.fc4.ModelFC4 import ModelFC4
from torch.nn.functional import normalize
from classes.fc4.repvit import utils
import numpy as np

from classes.core.Loss import Loss

def rec(u, v):
  ru = u[0]
  gu = u[1]
  bu = u[2]
  rv = v[0]
  gv = v[1]
  bv = v[2]
  cosines = (ru*rv+gu*gv+bu*bv)/(math.sqrt(ru*ru+gu*gu+bu*bu)*math.sqrt(rv*rv+gv*gv+bv*bv))
  angular_error = 180 * float(math.acos(max(-1, min(cosines, 1)))) / 3.141592653589793
  return angular_error

# def rep(u, v):
#   cosines = torch.sum(torch.div(u, v))/math.sqrt(3)
#   rep_error = 180 * torch.acos(torch.clamp(cosines, -1, 1)) / 3.141592653589793
#   return rep_error

class AngularLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = (torch.acos(dot) * torch.tensor(180 / math.pi)).to(self._device)

        # output =  torch.mean(angle).to(self._device)
        return angle

def rep(u, v):
  ru = u[0]
  gu = u[1]
  bu = u[2]
  rv = v[0]
  gv = v[1]
  bv = v[2]
  cosines = (ru/rv+gu/gv+bu/bv)/math.sqrt((math.pow(ru/rv, 2)+math.pow(gu/gv, 2)+math.pow(bu/bv, 2))*3)
  #rep_error = 180 * torch.acos(torch.clamp(cosines, -1, 1)) / 3.141592653589793
  rep_error = 180 * float(math.acos(max(-1, min(cosines, 1)))) / 3.141592653589793
  return rep_error

def arc(target):
  rt = target[0]
  gt = target[1]
  bt = target[2]
  xt = math.acos((rt+gt+bt)/math.sqrt(3*(rt*rt+gt*gt+bt*bt)))/math.sqrt(math.pow(2*rt-gt-bt,2)+3*math.pow(gt-bt, 2))*(2*rt-gt-bt)
  yt = math.acos((rt+gt+bt)/math.sqrt(3*(rt*rt+gt*gt+bt*bt)))/math.sqrt(math.pow(2*rt-gt-bt,2)+3*math.pow(gt-bt, 2))*math.sqrt(3)*(gt-bt)
  return xt, yt

#MIC
def MIC(seq, ns):
  mic = []
  for i in range(0, ns-1):
    u = seq[i]
    v = seq[i+1]
    mic.append(rec(u, v))
  return max(mic)
      
#STD
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


#python ./test/test.py

MODEL_PTH = './models/logs/mate30-new-0.0001'
MODEL_TYPE = "ctanet"
DATA_FOLDER = "mate30"
PATH_TO_LOGS = os.path.join("test", "logs")

def main(opt):
    model_type = opt.model_type
    data_folder = opt.data_folder
    loss_compute = AngularLoss(device=DEVICE)
    
    path_to_log = os.path.join(PATH_TO_LOGS, "{}_{}_{}".format(model_type, data_folder, time()))
    os.makedirs(path_to_log)

    path_to_pth = './models/logs/mate30-new-0.0001'
    evaluator = Evaluator()
    eval_data = {"file_names": [], "preds": [], "ground_truths": []}
    inference_times = []

    #1.mate30 2.P30pro 3.iphone 4.vivo 5.Xiaomi11 6.Xiaomi13(only test)
    test_set = CtaDataset(mode="test", device=1)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=20)
    print('Test set size: {}'.format(len(test_set)))

    model = ModelFC4()
    utils.replace_batchnorm(model)
        
    #val_loss = LossTracker()
    print('\n Loading pretrained {} model stored at: {} \n'.format(model_type, path_to_pth))
    model.load(path_to_pth)
    model.evaluation_mode()
    print("\n *** Testing model {} on {}\n".format(model_type, data_folder))
        

    with torch.no_grad():
        for i, (img, mimic, label, file_name) in enumerate(test_loader):
            img, mimic, label = img.to(DEVICE), mimic.to(DEVICE), label.to(DEVICE)
            tic = perf_counter()
            pred = model.predict(img, return_steps=False)
            toc = perf_counter()
            inference_times.append(toc - tic)
            "compute all loss"
            loss = loss_compute(pred, label)
            for ii in range(loss.shape[0]):
                evaluator.add_error(loss[ii].item())
                eval_data["file_names"].append(file_name[ii])
                eval_data["preds"].append(pred[ii].cpu().numpy())
                eval_data["ground_truths"].append(label[ii].cpu().numpy())

            if i % 10 == 0:
                print("Item {}: {}, AE: {:.4f}".format(i, file_name[0].split(os.sep)[-1], loss[0]))

    print(" \n Average inference time: {:.4f} \n".format(np.mean(inference_times)))

    eval_data["errors"] = evaluator.get_errors()
    metrics = evaluator.compute_metrics()

    pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(os.path.join(path_to_log, "metrics.csv"), index=False)
    pd.DataFrame(eval_data).to_csv(os.path.join(path_to_log, "eval.csv"), index=False)

    #计算mic,std,ae等指标
    camera = 0
    dataset_device = ['HuaweiMate30', 'HuaweiP30PRO', 'iphone14pm', 'vivoiqooneo5', 'Xiaomi11PRO', 'Xiaomi13', 'bcc']
    num_device = ['mate30', 'P30pro', 'iphonepm', 'vivo', 'xiaomi11pro', 'xiaomi13', 'bcc']
    all = {'mate30': [], 'P30pro': [], 'iphonepm': [], 'vivo': [], 'xiaomi11pro': [], 'xiaomi13': [], 'bcc':[]}

    eval_data1 = {"file_names": [], "mic": [], "std": [], "ae": []}

    df = pd.read_csv(os.path.join(path_to_log, "eval.csv"))
    files = df['file_names'].values.tolist()
    predss = df['preds'].values.tolist()
    gtss = df['ground_truths'].values.tolist()

    x = []
    y = []
    z = []
    seqs = []


    test_path = './dataset/CTA/test_'+num_device[camera]+'.npy'
    if camera == 6:
        test_path = './dataset/CTA/test_bcc.npy'
    test_info = np.load(test_path, allow_pickle=True).item()
    seqs = test_info['id']

    for seq in seqs:
        aes = []
        names = []
        seq_all = []
        for j in range(len(files)):
            #path_to_frame = str(files[j].split('/')[-2])
            seq_num = str(files[j].split('/')[-1].split(',')[0])
            if camera == 6:
                seq_num = str(files[j].split('test')[-1].split('_')[0])
            if str(seq) == seq_num and dataset_device[camera] in files[j]:
                names.append(files[j])
                #print(files[j])
                preds = predss[j][2:-2].split(' ')
                pred = []
                for p in preds:
                    if p != '':
                        pred.append(float(p))
                pred = [pred[0]/pred[1], 1.0, pred[2]/pred[1]]
                gts = gtss[j][2:-2].split(' ')
                gt = []
                for g in gts:
                    if g != '':
                        gt.append(float(g))
                #print(gt)
                aes.append(rec(pred, gt))
                seq_all.append(pred)
        ns = len(names)
        mic = MIC(seq_all, ns)
        std = STD(seq_all, ns)
        sae = sum(aes)/ns

        print(mic, std, sae)
        x.append(mic)
        y.append(std)
        z.append(sae)
        eval_data1["file_names"].append(seq)
        eval_data1["mic"].append(mic)
        eval_data1["std"].append(std)
        eval_data1["ae"].append(sae)


    all[num_device[camera]].append(sum(eval_data1["ae"])/len(seqs))
    all[num_device[camera]].append(sum(eval_data1["mic"])/len(seqs))
    all[num_device[camera]].append(sum(eval_data1["std"])/len(seqs))

    pd.DataFrame(eval_data1).to_csv(os.path.join(path_to_log, num_device[camera]+"_mic.csv"), index=False)  
    pd.DataFrame(all[num_device[camera]]).to_csv(os.path.join(path_to_log, num_device[camera]+"_all.csv"), index=False)
    print(all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_pth", type=str, default=MODEL_PTH)
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE)
    parser.add_argument('--data_folder', type=str, default=DATA_FOLDER)
    #parser.add_argument('--split_folder', type=str, default=SPLIT_FOLDER)
    opt = parser.parse_args()
    main(opt)
