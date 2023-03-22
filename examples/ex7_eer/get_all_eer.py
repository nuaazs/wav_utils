import os
import numpy as np
import paddle
import metrics

import torch
from tqdm import tqdm
import cfg

if __name__ == "__main__":
    embeddings = {}
    npy_path = f"../cache/{cfg.NAME}"
    # glob all score npy files
    score_npy_files = sorted([os.path.join(npy_path, x) for x in os.listdir(npy_path) if "scores" in x])
    labels_npy_files = sorted([os.path.join(npy_path, x) for x in os.listdir(npy_path) if "labels" in x])
    scores = []
    labels = []

    for score_npy,label_npy in zip(score_npy_files,labels_npy_files):
        print(f"Loading {score_npy}...")
        print(f"Loading {label_npy}...")
        scores += np.load(score_npy).tolist()
        labels += np.load(label_npy).tolist()

    labels = np.array(labels)
    scores = np.array(scores)
    print(f"labels: {labels.shape}")
    print(f"scores: {scores.shape}")
    # computer EER
    result = metrics.compute_eer(scores, labels,det_pic_save_path=f"./output_pngs/{cfg.NAME}_det.png",roc_pic_save_path=f"./output_pngs/{cfg.NAME}_roc.png")
    min_dcf = metrics.compute_min_dcf(result.fr, result.fa)
    print(f"EER: {result.eer}")
    print(f"minDCF: {min_dcf}")
    print(f"thresh: {result.thresh}")


