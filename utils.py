import os
import sys

ncore = '10'
os.environ["OMP_NUM_THREADS"] = ncore 
os.environ["OPENBLAS_NUM_THREADS"] = ncore 
os.environ["MKL_NUM_THREADS"] = ncore 
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore 
os.environ["NUMEXPR_NUM_THREADS"] = ncore 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import glob
import re
from itertools import chain, repeat
import datetime
import time
import pickle
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import random
import multiprocessing
from scipy.stats import zscore, rankdata
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='7'

print(device_lib.list_local_devices())

###========================================================
### FUNCTIONS for learning 
###========================================================

def get_initial_PEM_weight(tidl, expdata):
    tid = tidl.UniProt.unique().item()		# target list
    cidl = tidl.CID  						# compound list
    exp = expdata[expdata.index.isin(cidl)]
    # get initial weight by calculation of mean value
    exp_sum = exp.mean().fillna(0)
    initial_PEM = pd.DataFrame({'target_protein' : tid, 'Value' : [exp_sum.tolist()]})
    return initial_PEM

def sigmoid(z):
    return 1/(1+np.exp(-z))

def build_model(init_weight, args, **kwargs):
    # create model
    model = Sequential()
    model.add(Dense(y.shape[1], use_bias=False, input_shape = (X.shape[1],), activation='linear'))
    model.set_weights([np.array(init_weight)])          # weights 
    model.add(Dropout(args.dropout))             
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = args.lr))
    print(model.summary())
    return model

def cal_auc(mat):
    if len(set(mat.true)) == 2:
        fpr, tpr, _ = metrics.roc_curve(mat.true,  mat.prediction)
        aucval = metrics.roc_auc_score(mat.true, mat.prediction)
    else:
        fpr, tpr, aucval = None, None, None
    return {'CID':mat.idd.unique().item(),'Len' : len(mat[mat.true == 1]), 'fpr' : fpr.tolist(), 'tpr' : tpr.tolist(), 'AUC' : aucval}

def cal_prauc(mat):
    if len(set(mat.true)) == 2:
        tt = mat[mat.true == 1]
        if len(tt) * 10 < len(mat[mat.true == 0]):
            nt = mat[mat.true == 0].sample(len(tt) * 10)
        else:
            nt = mat[mat.true == 0]
        ttt = pd.concat([tt, nt])
        pr, rc, _ = metrics.precision_recall_curve(ttt.true, ttt.prediction)
        res = pd.DataFrame({'precision' : pr, 'recall' : rc})
        prauc_res = {'CID' : mat.idd.unique().item(), 'Len' : len(mat[mat.true == 1]), 'PRAUC' : metrics.auc(rc, pr), 'precision' : res.precision.tolist(), 'recall' : res.recall.tolist()}
    else:
        prauc_res = None
    return prauc_res

def validate_by_target_prediction(loss, x_test):
    loss = loss.T
    loss['idd'] = loss.index
    loss_melt = pd.melt(loss, id_vars = ['idd']).rename(columns = {'target_protein' : 'variable', 'value' : 'prediction'})
    x_test['idd'] = x_test.index
    true_dti_melt = pd.melt(x_test, id_vars = ['idd']).rename(columns = {'UniProt' : 'variable', 'value' : 'true'})
    loss_melt['idd'] = loss_melt.idd.astype(int)
    true_dti_melt['idd'] = true_dti_melt.idd.astype(int)
    auc_mat = pd.merge(loss_melt, true_dti_melt, on = ['idd', 'variable'], how  = 'inner').dropna()
    auc_mat_CIDl = [y for x, y in auc_mat.groupby('idd', as_index=False)]
    prauc_per_compound = list(map(cal_prauc, auc_mat_CIDl))
    prauc_per_compound = pd.DataFrame([x for x in prauc_per_compound if x is not None])
    auc_per_compound = pd.dataFrame(list(map(cal_auc, auc_mat_CIDl)))
    return prauc_per_compound, auc_per_compound
