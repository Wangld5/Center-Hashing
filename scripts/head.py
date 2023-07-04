from dataset.online_loader import *
from torchvision import models
from model.Net import Binary_hash, ResNet, MoCo
import os
import torch.optim as optim
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.nn.modules import BCELoss
from loguru import logger
import torch
torch.multiprocessing.set_sharing_strategy('file_system')  # multiprocessing to read files
import torch.nn as nn
import random
import shutil
from torch.utils.tensorboard import SummaryWriter
from loss.WeightLoss import *
from loss.CSQLoss import *
from loss.HashNetLoss import *
from loss.DSHLoss import *
from loss.DHN_Loss import *
from loss.CNNH import *
from loss.GreedyHash_Loss import *
from loss.DPNLoss import *
from loss.ourLoss import *
from loss.DPSH import *
from loss.DTSH import *
from scripts.utils import *
import pdb

