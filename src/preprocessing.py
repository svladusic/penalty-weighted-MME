import os
import xarray as xr
import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_monthly_flattened_model(models):
    np_array_models = []
    mu_annual_vals = []

    for model in models:
        tas_vals = model.tas.values
        con_tas = np.concatenate(tas_vals, axis=0)
        tas_flat = con_tas.flatten()
        np_array_models += [tas_flat]
        
        mu_monthly = np.mean(tas_vals, axis=(1,2))
        mu_annual = np.mean(mu_monthly.reshape(-1, 12), axis=1)
        mu_annual_vals += [mu_annual]

    return np_array_models, mu_annual_vals


def get_annual_flattened_model(models, reanal=[]):
    np_array_models = []
    for model in models:
        tas_vals = model.tas.values
        n = np.shape(tas_vals)[0]
        ub = int(n/12)
        mu_m = np.array([np.mean(tas_vals[i:i+12,:,:],axis=0).flatten()\
                         for i in range(ub)]).flatten()
        np_array_models += [mu_m]
    return np_array_models


def get_train_test_sets(inp_array, tp=0.2, vp=0.1):
    n = np.shape(inp_array)[1]
    tr_ub = int((1-tp-vp)*n)
    val_ub = int(tr_ub + vp*n)
    np.random.shuffle(inp_array.T)
    train_set = inp_array[:, :tr_ub]
    val_set = inp_array[:, tr_ub:val_ub]
    test_set = inp_array[:, val_ub:]
    return train_set, val_set, test_set

if __name__ == "__main__":    
    CMIP_dir = "../data/CMIP6 TAS" 
    cutoff_1980 = 12*16*(1980 - 1850)
    cutoff_2014 = 12*16*(2014 - 1850 + 1)
    
    model_names_raw = os.listdir(CMIP_dir)
    model_names = [mn.split('_')[3] for mn in model_names_raw]
    np.savetxt("model_names.csv", model_names, delimiter=",",  fmt='%s')
    models = [xr.open_dataset(CMIP_dir + mn, decode_times=False)    
              for mn in model_names_raw]
    
    
# =============================================================================
#     Generate and Export Monthly Data
# =============================================================================
    tas_flat, mu_annual = get_monthly_flattened_model(models[:-1])
    tas_flat = tas_flat[:24] + tas_flat[25:]
    mu_annual = np.vstack(mu_annual[:24] + mu_annual[25:])
    mu_annual_proj = mu_annual[:, 164:]
    np.savetxt("mu_ann_proj.csv", mu_annual_proj, delimiter=",")
    tas_flat = np.vstack(tas_flat)
    hist_mtas = tas_flat[:, cutoff_1980:cutoff_2014]
    reanal = xr.open_dataset(CMIP_dir + 'merra_ts_144_90.nc',\
                             decode_times=False)         
    hist_tas_reanal = reanal.ts.values[0:420, :, :].flatten()
    
    hist_tas = np.vstack([hist_mtas, hist_tas_reanal])
    pw_mdist_mat = squareform(pdist(hist_mtas, metric='euclidean'))
    
    np.savetxt("hist_mtas.csv", hist_mtas, delimiter=",")
    np.savetxt("pw_mdist_mat.csv", pw_mdist_mat, delimiter=",")
    np.savetxt("hist_tas.csv", hist_tas, delimiter=",")
    
    hist_tas_stnd = (hist_tas - np.mean(hist_tas, axis=0))/np.std(hist_tas)
    hist_mtas_stnd = hist_tas_stnd[:-1,:-1]
    pw_mdist_mat_stnd = squareform(pdist(hist_mtas_stnd, metric='euclidean'))
    
    np.savetxt("hist_mtas_stnd.csv", hist_mtas_stnd, delimiter=",")
    np.savetxt("pw_mdist_mat_stnd.csv", pw_mdist_mat_stnd, delimiter=",")
    np.savetxt("hist_tas_stnd.csv", hist_tas_stnd, delimiter=",")

    ht_trs, ht_vs, ht_ts = get_train_test_sets(hist_tas)
    
    np.savetxt("X_tr.csv",ht_trs[:-1,:], delimiter=",")
    np.savetxt("y_tr.csv", ht_trs[-1, :], delimiter=",")
    np.savetxt("X_val.csv",ht_vs[:-1,:], delimiter=",")
    np.savetxt("y_val.csv", ht_vs[-1, :], delimiter=",")
    np.savetxt("X_test.csv", ht_ts[:-1,:], delimiter=",")
    np.savetxt("y_test.csv", ht_ts[-1,:], delimiter=",")
        
    ht_trs_stnd, ht_vs_stnd, ht_ts_stnd = get_train_test_sets(hist_tas_stnd)

    np.savetxt("X_tr_stnd.csv", ht_trs_stnd[:-1, :], delimiter=",")
    np.savetxt("y_tr_stnd.csv", ht_trs_stnd[-1, :], delimiter=",")
    np.savetxt("X_val_stnd.csv", ht_vs_stnd[:-1, :], delimiter=",")
    np.savetxt("y_val_stnd.csv", ht_vs_stnd[-1, :], delimiter=",")
    np.savetxt("X_test_stnd.csv", ht_ts_stnd[:-1, :], delimiter=",")
    np.savetxt("y_test_stnd.csv", ht_ts_stnd[-1, :], delimiter=",")
    
# =============================================================================
#     Generate and Export Yearly Data
# =============================================================================
    cutoff_1980 = 16*(1980 - 1850)
    cutoff_2014 = 16*(2014 - 1850 + 1)
    tas_ann = get_annual_flattened_model(models[:-1])
    tas_ann = tas_ann[:24] + tas_ann[25:] 
    tas_ann = np.vstack(tas_ann)
    tas_ann_model = tas_ann[:, cutoff_1980:cutoff_2014]
    hist_tas_reanal_ann = [np.mean(reanal.ts.values[12*i:12*i+12, :, :], \
                                   axis=0).flatten() for i in range(35)]
    hist_tas_reanal_ann = np.array(hist_tas_reanal_ann).flatten()

    hist_tas_ann = np.vstack([tas_ann_model, hist_tas_reanal_ann])
    pw_mdist_mat = squareform(pdist(tas_ann_model, metric='euclidean'))
    np.savetxt("hist_mtas_ann.csv", tas_ann_model, delimiter=",")
    np.savetxt("pw_mdist_mat_ann.csv", pw_mdist_mat, delimiter=",")
    np.savetxt("hist_tas_ann.csv", hist_tas_ann, delimiter=",")
    
    hist_tas_ann_stnd = (hist_tas_ann - \
                         np.mean(hist_tas_ann,axis=0))/np.std(hist_tas_ann)
    hist_mtas_ann_stnd = hist_tas_stnd[:-1,:]
    pw_mdist_mat_ann_stnd = squareform(pdist(hist_mtas_ann_stnd,\
                                             metric='euclidean'))
        
    np.savetxt("hist_mtas_ann_stnd.csv", hist_mtas_ann_stnd, delimiter=",")
    np.savetxt("pw_mdist_mat_ann_stnd.csv", pw_mdist_mat_ann_stnd,\
               delimiter=",")
    np.savetxt("hist_tas_ann_stnd.csv", hist_tas_ann_stnd, delimiter=",")

    ht_trs_ann, ht_vs_ann, ht_ts_ann = get_train_test_sets(hist_tas_ann)

    np.savetxt("X_tr_ann.csv", ht_trs_ann[:-1, :], delimiter=",")
    np.savetxt("y_tr_ann.csv", ht_trs_ann[-1, :], delimiter=",")
    np.savetxt("X_val_ann.csv", ht_vs_ann[:-1, :], delimiter=",")
    np.savetxt("y_val_ann.csv", ht_vs_ann[-1, :], delimiter=",")
    np.savetxt("X_test_ann.csv", ht_ts_ann[:-1, :], delimiter=",")
    np.savetxt("y_test_ann.csv", ht_ts_ann[-1, :], delimiter=",")
    
    ht_trs_ann_stnd, ht_vs_ann_stnd, ht_ts_ann_stnd =\
        get_train_test_sets(hist_tas_ann_stnd)

    np.savetxt("X_tr_ann_stnd.csv", ht_trs_ann_stnd[:-1, :],\
               delimiter=",")
    np.savetxt("y_tr_ann_stnd.csv", ht_trs_ann_stnd[-1, :],\
               delimiter=",")
    np.savetxt("X_val_ann_stnd.csv", ht_vs_ann_stnd[:-1, :],\
               delimiter=",")
    np.savetxt("y_val_ann_stnd.csv", ht_vs_ann_stnd[-1, :],\
               delimiter=",")    
    np.savetxt("X_test_ann_stnd.csv", ht_ts_ann_stnd[:-1, :],\
               delimiter=",")
    np.savetxt("y_test_ann_stnd.csv", ht_ts_ann_stnd[-1, :],\
               delimiter=",")
