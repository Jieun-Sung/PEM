import os
import sys
import argparse
import numpy as np
import pandas as pd
from utils import *

###========================================================
### DIRECTORY & PARAMETERS
###========================================================
datadir = '/'
outputpath = datadir + '/Learning'

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

#----------------------------------------------------------
parser = argparse.ArgumentParser()
args = parser.parse_args("")

# ==== Model Architecture Config ==== #
args.dose = 100
args.cell_line = 'MCF7'

args.test_size = 0.2   
args.lr = 1e-4

args.random_seed = 31
args.num_epochs = 260
args.dropout = 0.0

###========================================================
### LOAD DATA 
###========================================================

inputdata = pd.read_csv(f'{datadir}/KMAP_{cell_line}_{dose}_inputdata.csv', index_col = 0)
realdata = pd.read_csv(f'{datadir}/KMAP_{cell_line}_{dose}_norm_deltazscore.csv', index_col = 0)
dti = pd.read_csv(datadir + '/DTI_data.tsv', sep = '\t')

X = inputdata
y = realdata

X_train, X_test = train_test_split(X, test_size = args.test_size, random_state=args.random_seed)
y_train = y.loc[X_train.index,:]
y_test = y.loc[X_test.index,:]

###========================================================
### Get initial PEM with train dataset
###========================================================

dti_split_by_target = [y for x, y in dti.groupby('UniProt', as_index=False)]

temp = pd.concat([x for x in list(map(get_consensus_gene_per_targets, dti_split_by_target, repeat(y_train)))])
init_weight_df = init_weight.Value.apply(pd.Series)
init_weight_df.index = init_weight.target_protein
init_weight_df.columns = y_train.columns
init_weight_df = init_weight_df.fillna(0)

init_weight = sigmoid(init_weight)

###========================================================
### Matrix Factorization
###========================================================

X = inputdata
y = realdata

final_mm = build_model(init_weight, args)
history = final_mm.fit(X_train, y_train, epochs = args.num_epochs, verbose=1)

###========================================================
### Target prediction 
###========================================================

pem_res = pd.DataFrame(final_mm.layers[0].get_weights()[0], index = init_weight.index, columns = init_weight.columns)

from numpy.linalg import norm
cossim = pd.DataFrame()
for i in range(len(y_test)):
    l2 = np.array(y_test.iloc[i,:])
    ll = pem_res.apply(lambda x: np.dot(np.array(x), l2) / (norm(np.array(x)) * norm(l2)), axis = 1)
    cossim = pd.concat([cossim, ll], axis = 1)

cossim.columns = y_test.index

prauc_per_compound, auc_per_compound = cal_auc_all(cossim, X_test)
