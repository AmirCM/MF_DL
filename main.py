import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2 as cv
import numpy as np

IS_Rebuild = 0


class CleanVsBlurred:
    CLEAN = 'PET'
    BLURRED = 'blurred'
    LABELS = {BLURRED: 0, CLEAN: 1}
    training_data = []

    def make_training_data(self):
        mat_content = sio.loadmat('data.mat')
        print(mat_content.keys())

        for lbl in self.LABELS:
            for i in tqdm(range(256)):
                img = mat_content[lbl][:, :, i]
                self.training_data.append([img, np.eye(2)[self.LABELS[lbl]]])
                cv.imshow(lbl, img)
                cv.waitKey(1)
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)


if __name__ == "__main__":
    clean_vs_blurred = CleanVsBlurred()
    if IS_Rebuild:
        clean_vs_blurred.make_training_data()
    else:
        clean_vs_blurred.training_data = np.load("training_data.npy", allow_pickle=True)
