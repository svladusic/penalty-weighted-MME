import os
import numpy as np
import matplotlib.pyplot as plt
from utils import GD

plt.rcParams.update({'font.size': 22})

os.chdir("../data/CSVs")

def get_normalized_weights(ws):
    return np.abs(ws)/np.sum(np.abs(ws))

def plot_errs(tsl, vsl, th, reg_param=None, loss_type=None):
    if not loss_type:
        loss_type = 'Pointwise'
    if not reg_param:
        reg_param = '1'
    tsl = tsl[:-th]
    vsl = vsl[:-th]
    n = int(np.shape(tsl)[0])
    epochs = np.arange(1, n+1)
    plt.figure(figsize=(10.80,10.80))
    plt.plot(epochs, tsl, label='Training loss')
    plt.plot(epochs, vsl, label='Validation loss')
    plt.grid()
    plt.title(f'RMSE Loss vs Epoch with Reg '
              f'Param {reg_param} and {loss_type} Penalty Function')
    plt.ylabel('RMSE Loss')
    plt.xlabel('Epoch Number')
    plt.legend()
    return None


def compare_costs(tl1, tl2, th, param=None):
    no_epochs = np.max([np.size(tl1), np.size(tl2)])
    epochs = np.arange(no_epochs)    
    
    train_sets = [tl1, tl2]
    for i, ds in enumerate(train_sets):
        if np.size(ds) < no_epochs:
            endpts = ds[-th]*np.ones(no_epochs - np.size(ds))
            train_sets[i] = np.concatenate([ds, endpts])
    
    plt.figure(figsize=(10.80,10.80))
    plt.plot(epochs, train_sets[0], label='Training loss Monthly')
    plt.plot(epochs, train_sets[1], label='Training loss Annual')
    plt.grid()
    if not param:
        plt.title('Monthly and Annual Loss vs Epoch (NNLS)')
    else:
        plt.title('Monthly and Annual Loss vs Epoch (Family w/ Param 10)')
    plt.ylabel('RMSE Loss')
    plt.xlabel('Epoch Number')
    plt.legend()    



def plot_weights(ind_mn, ws, reg_param=None, loss_type=None):
    if not loss_type:
        loss_type = 'Pointwise'
    if not reg_param:
        reg_param = '1'
    
    plt.figure(figsize=(19.20,10.80))
    plt.scatter(np.arange(29), np.flip(np.sort(ws)), marker = 'x', color='k',\
                s=120)
    plt.plot(np.arange(29), np.ones(29)/29, 'r--', label='Equal Weights')
    plt.xticks(ticks=np.arange(29), labels=ind_mn, rotation=90)
    plt.grid()
    plt.title(f'Model Weights '
              f'given Param {reg_param} and {loss_type} Penalty Function')
    plt.ylabel('Weights')
    plt.legend()
    return None


def plot_tas_proj(mu_annual_proj, ws, reg_param=None, loss_type=None):
    mu_annual_proj = mu_annual_proj - \
                     np.expand_dims(mu_annual_proj[:, 0], axis=1)
    if not loss_type:
        loss_type = 'Pointwise'
    if not reg_param:
        reg_param = '1'
    proj_uw = np.average(mu_annual_proj, axis=0)
    proj_w = np.average(mu_annual_proj, axis=0, weights = ws)
    x_vals = np.arange(2015, 2015 + np.size(proj_uw))
    
    plt.figure(figsize=(10.80,10.80))
    plt.plot(x_vals, proj_uw, color='orange',\
             label = 'Unweighted CMIP6 projection')
    plt.plot(x_vals, proj_w, color='b', label = 'Weighted CMIP6 projection')
    plt.grid()
    plt.title(f'CMIP6 Projections '
              f'with Param {reg_param} and {loss_type} Penalty Function')
    plt.ylabel('Temperature relative to 2014')
    plt.xlabel('Year')
    plt.legend()
    return None


model_names = ['ACCESS-CM2',
                'ACCESS-ESM1-5',
                'AWI-CM-1-1-MR',
                'BCC-CSM2-MR',
                'CAMS-CSM1-0',
                'CanESM5-CanOE-p2',
                'CanESM5-p1',
                'CESM2-WACCM',
                'CESM2',
                'CNRM-CM6-1-HR-f2',
                'CNRM-ESM2-1-f2',
                'EC-Earth3-Veg',
                'EC-Earth3',
                'FGOALS-f3-L',
                'FIO-ESM-2-0',
                'GISS-E2-1-G-p3',
                'HadGEM3-GC31-LL-f3',
                'INM-CM4-8',
                'INM-CM5-0',
                'IPSL-CM6A-LR',
                'KACE-1-0-G',
                'MCM-UA-1-0',
                'MIROC-ES2L-f2',
                'MIROC6',
                'MPI-ESM1-2-HR',
                'MPI-ESM1-2-LR',
                'MRI-ESM2-0',
                'NESM3',
                'NorESM2-MM',
                'UKESM1-0-LL-f2',
                '90.nc']

model_families = [[2, 24, 26], 
                  [7, 8, 27], 
                  [0, 1, 16, 20, 28],
                  [9, 10, 11, 12],
                  [17, 18],
                  [5, 6],
                  [22, 23],
                  [3],
                  [4],
                  [13],
                  [14],
                  [15],
                  [19],
                  [21],
                  [25]]

indices = np.arange(31)

model_dict = dict(zip(indices, model_names))
mu_annual_proj = np.genfromtxt('mu_ann_proj.csv', delimiter=',')
# =============================================================================
# Monthly Datasets
# =============================================================================

X_tr = np.genfromtxt('X_tr.csv', delimiter=',')
y_tr = np.genfromtxt('y_tr.csv', delimiter=',')
X_val = np.genfromtxt('X_val.csv', delimiter=',')
y_val = np.genfromtxt('y_val.csv', delimiter=',')
X_test = np.genfromtxt('X_test.csv', delimiter=',')
y_test = np.genfromtxt('y_test.csv', delimiter=',')
hist_mtas = np.genfromtxt('hist_mtas.csv', delimiter=',')
pw_mdist_mat = np.genfromtxt('pw_mdist_mat.csv', delimiter=',')
    
X_tr_stnd = np.genfromtxt('X_tr_stnd.csv', delimiter=',')
y_tr_stnd = np.genfromtxt('y_tr_stnd.csv', delimiter=',')
X_val_stnd = np.genfromtxt('X_val_stnd.csv', delimiter=',')
y_val_stnd = np.genfromtxt('y_val_stnd.csv', delimiter=',')
X_test_stnd = np.genfromtxt('X_test_stnd.csv', delimiter=',')
y_test_stnd = np.genfromtxt('y_test_stnd.csv', delimiter=',')
hist_mtas_stnd = np.genfromtxt('hist_mtas_stnd.csv', delimiter=',')
pw_mdist_mat_stnd = np.genfromtxt('pw_mdist_mat_stnd.csv', delimiter=',')

# =============================================================================
# Annual Datasets
# =============================================================================

X_tr_ann = np.genfromtxt('X_tr_ann.csv', delimiter=',')
y_val_ann = np.genfromtxt('y_val_ann.csv', delimiter=',')
X_val_ann = np.genfromtxt('X_val_ann.csv', delimiter=',')
y_tr_ann = np.genfromtxt('y_tr_ann.csv', delimiter=',')
X_test_ann = np.genfromtxt('X_test_ann.csv', delimiter=',')
y_test_ann = np.genfromtxt('y_test_ann.csv', delimiter=',')
hist_mtas_ann = np.genfromtxt('hist_mtas_ann.csv', delimiter=',')
pw_mdist_mat_ann = np.genfromtxt('pw_mdist_mat_ann.csv', delimiter=',')
    
X_tr_ann_stnd = np.genfromtxt('X_tr_ann_stnd.csv', delimiter=',')
y_tr_ann_stnd = np.genfromtxt('y_tr_ann_stnd.csv', delimiter=',')
X_val_ann_stnd = np.genfromtxt('X_val_ann_stnd.csv', delimiter=',')
y_val_ann_stnd = np.genfromtxt('y_val_ann_stnd.csv', delimiter=',')
X_test_ann_stnd = np.genfromtxt('X_test_ann_stnd.csv', delimiter=',')
y_test_ann_stnd = np.genfromtxt('y_test_ann_stnd.csv', delimiter=',')
hist_mtas_ann_stnd = np.genfromtxt('hist_mtas_ann_stnd.csv', delimiter=',')
pw_mdist_mat_ann_stnd = np.genfromtxt('pw_mdist_mat_ann_stnd.csv', delimiter=',')


# =============================================================================
# Check with NNLS
# =============================================================================

err_month = nnls(X_tr_stnd.T, y_tr_stnd)[1]
err_yearly = nnls(X_tr_ann_stnd.T, y_tr_ann_stnd)[1]

# We can see that yearly is better.
 
# =============================================================================
# Begin Gradient Descent for NNLS and Compare Cost Functions
# =============================================================================

ws_unif = np.ones(29)/29
reg_params = [1, 10, 100]
loss_types = ['pw', 'fam']
th = 5

wsm, tlm, vallm, testlm = GD(y_tr_stnd, X_tr_stnd, y_val_stnd,\
                     X_val_stnd, y_test_stnd, X_test_stnd,\
                     ws_unif, pw_mdist_mat_stnd, 1e-4, 0, 0, th, 'pw')

ws, tl, vall, testl = GD(y_tr_ann_stnd, X_tr_ann_stnd, y_val_ann_stnd,\
                     X_val_ann_stnd, y_test_ann_stnd, X_test_ann_stnd,\
                     ws_unif, pw_mdist_mat_ann_stnd, 1e-4, 0, 0, th, 'pw')

compare_costs(tlm, tl, th)
print(testlm)
print(testl)


wsm, tlm_10, vallm, testlm = GD(y_tr_stnd, X_tr_stnd, y_val_stnd,\
                     X_val_stnd, y_test_stnd, X_test_stnd,\
                     ws_unif, model_families, 1e-4, 10, 10, th, 'fam')

ws, tl_10, vall, testl = GD(y_tr_ann_stnd, X_tr_ann_stnd, y_val_ann_stnd,\
                     X_val_ann_stnd, y_test_ann_stnd, X_test_ann_stnd,\
                     ws_unif, model_families, 1e-4, 10, 10, th, 'fam')

compare_costs(tlm_10, tl_10, th, True)
print(testlm)
print(testl)

no_epochs = np.max([np.size(tlm), np.size(tl)])
epochs = np.arange(no_epochs)    

train_sets = [tlm, tl]
for i, ds in enumerate(train_sets):
    if np.size(ds) < no_epochs:
        endpts = ds[-1]*np.ones(no_epochs - np.size(ds))
        train_sets[i] = np.concatenate([ds, endpts])

plt.figure(figsize=(10.80,10.80))
plt.plot(epochs, train_sets[0], label='Training loss Monthly')
plt.plot(epochs, train_sets[1], label='Training loss Annual')
plt.grid()
plt.title('Training Set RMSE Loss vs Epoch for Monthly and Annual Datasets')
plt.ylabel('RMSE Loss')
plt.xlabel('Epoch Number')
plt.legend()

ws_n = get_normalized_weights(ws)
indices = np.flip(np.argsort(ws_n))
ind_mn = [model_names[i] for i in indices]

plt.figure(figsize=(19.20,10.80))
plt.scatter(np.arange(29), np.flip(np.sort(ws)), marker = 'x', color='k',\
            s=120)
plt.plot(np.arange(29), np.ones(29)/29, 'r--', label='Equal Weights')
plt.xticks(ticks=np.arange(29), labels=ind_mn, rotation=90)
plt.grid()
plt.title('Model Weights using NNLS and no Regularization Terms')
plt.ylabel('Weights')
plt.legend()

wn_pw = []
wn_fam = []
for l in reg_params:
    for t in loss_types:
        
        if t == 'pw':
            t_str = 'Pointwise'
            inp_array = pw_mdist_mat_ann_stnd 
        else:
            t_str = 'Family'
            inp_array = model_families
            
        l_str = str(l)
        ws, tl, vall, testl = GD(y_tr_ann_stnd, X_tr_ann_stnd, y_val_ann_stnd,\
                              X_val_ann_stnd, y_test_ann_stnd, X_test_ann_stnd,\
                              ws_unif, inp_array, 1e-4, l, 0, th, t)
        wn = np.sum(np.abs(ws))
        if t == 'pw':
            wn_pw += [wn]
        else:
            wn_fam += [wn]
        ws_n = get_normalized_weights(ws)
        indices = np.flip(np.argsort(ws_n))
        ind_mn = [model_names[i] for i in indices]
        wm_pairs = [str(ind_mn[i])+ ': ' + str(ws_n[i]) for i in indices]
        nl = '\n'
        
        print(f'Model Weights:{nl}{nl.join(wm_pairs)}')
        
        plot_errs(tl, vall, th, l_str, t_str)
        plot_weights(ind_mn, ws_n, l_str, t_str)
        plot_tas_proj(mu_annual_proj, ws_n, l_str, t_str)

print('\n\n')
print('Done!')

plt.figure(figsize=(10.80,10.80))
plt.plot([1, 10, 100], wn_pw, 'orange', label='Pointwise Penalty')
plt.plot([1, 10, 100], wn_fam, 'blue', label='Family Penalty')
plt.grid()
plt.title('1-Norm of Weight Vectors vs Hyper parameter values')
plt.xlabel('Hyperparameter value ($\lambda)$')
plt.ylabel('Weight Vector 1-Norm')
plt.grid()
plt.legend()