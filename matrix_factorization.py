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

dti_binary = pd.read_csv(f'{datadir}/KMAP_{cell_line}_{dose}_dti_binary.csv', index_col = 0)
norm_deltaz_data = pd.read_csv(f'{datadir}/KMAP_{cell_line}_{dose}_norm_deltazscore.csv', index_col = 0)
dti_matrix = pd.read_csv(datadir + '/DTI_data.tsv', sep = '\t')

X = dti_binary
y = norm_deltaz_data

X_train, X_test = train_test_split(X, test_size = args.test_size, random_state=args.random_seed)
y_train = y.loc[X_train.index,:]
y_test = y.loc[X_test.index,:]

###========================================================
### Get initial PEM with train dataset
###========================================================

dti_split_by_target = [y for x, y in dti_matrix.groupby('UniProt', as_index=False)]

init_weight = pd.concat([x for x in list(map(get_initial_PEM_weight, dti_split_by_target, repeat(y_train)))])
init_weight = init_weight.Value.apply(pd.Series)
init_weight.index = init_weight.target_protein
init_weight.columns = y_train.columns
init_weight = init_weight.fillna(0)
init_weight = sigmoid(init_weight)

###========================================================
### Matrix Factorization
###========================================================

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

prauc_per_compound, auc_per_compound = validate_by_target_prediction(cossim, X_test)
