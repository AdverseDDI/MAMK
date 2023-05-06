from matplotlib import pyplot  
import scipy as sp    
import cupy as np
import numpy as npp
import pandas as pd
from matplotlib import pylab  
from sklearn.datasets import load_files   
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  
from sklearn.linear_model import LogisticRegression  
import time 
from scipy.linalg.misc import norm
from copy import deepcopy
# from numpy import *
import os
np.set_printoptions(threshold=np.inf) 
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from sklearn import metrics 
import time
from sklearn.metrics import classification_report, average_precision_score
from sklearn.linear_model import LogisticRegression 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import pickle
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES']='1'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
npp.random.seed(1) # 用于numpy的随机数
np.random.seed(1)

class DeepNeuralNetworkForADDIModeling(nn.Module):
	def __init__(self, KernelVectorPQLength,
		AdverseInteractionLength):
		super(DeepNeuralNetworkForADDIModeling, self).__init__()
	# the design neural network by mapping the integration of common and specific attribute representations
	# i.e., the kernel vector by common attribute representation and specific attribute representation 
	# of adverse drug pair (d_i, d_j) into its adverse interaction vector r_{ij}^K with Sigmoid activation 
	# function and the nodes of hidden layers {384*2, 512, 512, 1024,1024, K}. ReLU  Sigmoid Tanh, Hardtanh, Softsign ELU LeakyReLU 
	#  CELU SELU GELU PReLU Softplus RReLU
		self.DeepNeuralNetwork=nn.Sequential(
			nn.Linear(KernelVectorPQLength,KernelVectorPQLength),nn.Dropout(0.5),
			nn.ELU (),
			nn.Linear(KernelVectorPQLength,1024),nn.Dropout(0.5),
			nn.ELU (),
			nn.Linear(1024,1024),nn.Dropout(0.5),
			nn.ELU (),
			nn.Linear(1024,2048),nn.Dropout(0.5),
			nn.ELU (),
			nn.Linear(2048,2048),nn.Dropout(0.5),
			nn.ELU (),				
    		nn.Linear(2048,AdverseInteractionLength),nn.Dropout(0.5),
			nn.ELU (),
			nn.Linear(AdverseInteractionLength,AdverseInteractionLength),nn.Dropout(0.5),
			nn.ELU (),
			)		
	def forward(self,KernelVectorPQ):
		PredictedKernelVectorPQ=self.DeepNeuralNetwork(KernelVectorPQ)
		return PredictedKernelVectorPQ
# This function is to compute the loss between the predicted Adverse interaction vector and 
# the ground-truth Adverse interaction vector of an Adverse drug pair. Here, we use MSELoss function 
# to calculate the loss. Besides, other loss function (e.g., BCELoss, BCEWithLogitsLoss) can also be used 
# for loss estimation
def LossFunctionOfPredictedAndGroundTruthAdverseInteractionVector(PredictedKernelVectorPQ,
	AdverseInteractionVector):
	Loss_Function=nn.SmoothL1Loss()
	# Loss_Function=nn.L1Loss()
	# Loss_Function=nn.MSELoss()
	# Loss_Function=nn.CrossEntropyLoss()

	# Loss_Function=nn.BCEWithLogitsLoss()
	PredictedKernelVectorPQ=PredictedKernelVectorPQ.unsqueeze(0)
	AdverseInteractionVector=AdverseInteractionVector.unsqueeze(0)
	# Initialize the MSELoss by nn.MSELoss()
	Loss=Loss_Function(PredictedKernelVectorPQ,AdverseInteractionVector)
	# calculate  the loss between the predicted Adverse interaction vector and 
	# the ground-truth Adverse interaction vector of an Adverse drug pair
	return Loss
	# return the Loss value