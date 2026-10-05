import os
import numpy as np
import torch
from tqdm import tqdm
import pickle


def normalize_pc(point_cloud):

    centroid = np.mean()
