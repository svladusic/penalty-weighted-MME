import numpy as np
from scipy.optimize import nnls

def get_pointwise_loss(ws, pw_dist_mat):
    W = np.outer(ws,ws)
    w_dist_mat = W*pw_dist_mat
    loss = np.sum(w_dist_mat)/2
    return loss


def get_pointwise_loss_grad(ws, pw_dist_mat):
    n = np.shape(pw_dist_mat)[0]
    loss = np.zeros(n)
    for i in range(n):
        loss[i] = np.sum(np.abs(np.delete(ws, i))/np.delete(pw_dist_mat[i,:], i))  
    return loss


def get_stdev_loss(ws, dist_array):
    w_norm = np.sum(ws)
    mu = np.mean(dist_array[-1], axis=0)
    dw = np.array([np.linalg.norm(d-mu) for d in dist_array])
    var_num = (np.sum(np.abs(ws)*dw))
    out = w_norm/var_num
    return out


def get_stdev_loss_grad(ws, dist_array):
    mu = np.mean(dist_array[-1], axis=0)
    dw = np.array([np.linalg.norm(d-mu) for d in dist_array])
    w_denom = (np.sum(np.abs(ws)*dw))
    w_num = np.sum(np.abs(ws))
    sws = np.sign(ws)
    loss_grad = sws*(1/w_denom - dw*w_num/(w_denom**2))
    return loss_grad


def get_family_loss(ws, families):
    loss = 0
    
    for f in families:
        loss += np.sum(ws[f])/np.sum(np.delete(ws, f))
    
    return loss        


def get_family_grad(ws, families):
    n = np.size(ws)
    grads = np.zeros(n)
    for k in range(n):
        for f in families:
            num = np.sum(ws[f])
            denom = np.sum(np.delete(ws, f))**2
            if k in f:
                grads[k] += 1
            else:
                grads[k] += -num/denom
    grads = grads*np.sign(ws)
    return grads


def get_OLS_loss(x, Y, ws):
    N = np.shape(Y)[1]
    return 1/(2*N)*np.linalg.norm(x-np.sum(np.abs(ws.T)@Y, axis=0))**2


def get_OLS_grad(x, Y, ws):
    N = np.shape(Y)[1]
    return -np.sign(ws)*N**(-1) * \
            np.sum(np.abs(x-np.sum(np.abs(ws.T)@Y, axis=0))*Y,axis=1)


def full_loss_pw(x, Y, ws, pw_dist_mat, l1, l2):
    return get_OLS_loss(x, Y, ws) + l1*get_pointwise_loss(ws, pw_dist_mat) + \
           l2*np.linalg.norm(ws)**2


def update_pwl_grad(x, Y, ws, pw_dist_mat, l1, l2):
    return get_OLS_grad(x, Y, ws) + \
           l1*get_pointwise_loss_grad(ws, pw_dist_mat) + l2*ws 


def full_loss_std(x, Y, ws, dist_array, l1, l2):
    return get_OLS_loss(x, Y, ws) + l1*get_stdev_loss(ws, dist_array) + \
           l2*np.linalg.norm(ws)**2


def update_std_grad(x, Y, ws, families, l1, l2):
    return get_OLS_grad(x, Y, ws) + \
           l1*get_family_grad(ws, families) + l2*ws 

def full_loss_fam(x, Y, ws, families, l1, l2):
    return get_OLS_loss(x, Y, ws) + l1*get_family_loss(ws, families) + \
           l2*np.linalg.norm(ws)**2


def update_fam_grad(x, Y, ws, families, l1, l2):
    return get_OLS_grad(x, Y, ws) + \
           l1*get_family_grad(ws, families) + l2*ws 
           
           
def get_full_loss(x, Y, ws, array, l1, l2, loss=None):
    if not loss:
        loss = 'pw'
    if loss == 'fam':
        loss = full_loss_fam(x, Y, ws, array, l1, l2)
    else:
        loss = full_loss_pw(x, Y, ws, array, l1, l2)
    return loss

def grad_desc(x, Y, ws, array, lr, l1, l2, loss=None):
    if not loss:
        loss = 'std'
    if loss == 'pw':
        ws = ws - lr*update_pwl_grad(x, Y, ws, array, l1, l2)
    else:
        ws = ws - lr*update_fam_grad(x, Y, ws, array, l1, l2)
    return ws


def GD(x_tr, Y_tr, x_val, Y_val, x_test, Y_test, ws, array, lr, l1, l2, th=10, loss=None):
    training_loss = []
    val_loss = []
    epoch = 0
    ind = 1
    ws_vals = []
    
    while ind:
        epoch += 1
        # print(f'Currenct Epoch: {epoch}')
        training_loss += [get_full_loss(x_tr, Y_tr, ws, array, l1, l2, loss)]
        val_loss += [get_full_loss(x_val, Y_val, ws, array, l1, l2, loss)]
        ws = grad_desc(x_tr, Y_tr, ws, array, lr, l1, l2, loss)
        ws_vals += [ws]
        # print(ws)
        if len(training_loss) > th:
            if np.all(training_loss[-2-(th+1):-2] < training_loss[-1]):
                ind = 0
    ws = ws_vals[-th] 
    RMSE_train = np.sqrt(training_loss)
    RMSE_val = np.sqrt(val_loss)
    RMSE_test = np.sqrt(get_full_loss(x_test, Y_test, ws, array, l1, l2, loss))
    return ws, RMSE_train, RMSE_val, RMSE_test

    
if __name__ == '__main__':
    # model_names = np.genfromtxt('model_names.csv', delimiter=',')
    X_tr = np.genfromtxt('X_tr.csv', delimiter=',')
    y_tr = np.genfromtxt('y_tr_stnd.csv', delimiter=',')
    X_test = np.genfromtxt('X_test.csv', delimiter=',')
    y_test = np.genfromtxt('y_test.csv', delimiter=',')
    hist_mtas = np.genfromtxt('hist_mtas.csv', delimiter=',')
    
    X_tr_stnd = np.genfromtxt('X_tr_stnd.csv', delimiter=',')
    y_tr_stnd = np.genfromtxt('y_tr_stnd.csv', delimiter=',')
    X_test_stnd = np.genfromtxt('X_test_stnd.csv', delimiter=',')
    y_test_stnd = np.genfromtxt('y_test_stnd.csv', delimiter=',')
    hist_mtas_stnd = np.genfromtxt('hist_mtas_stnd.csv', delimiter=',')
    
    
    pw_mdist_mat = np.genfromtxt('pw_mdist_mat.csv', delimiter=',')
    ws_unif = np.ones(29)/29
    ws_nnls_stnd = nnls(X_tr_stnd.T, y_tr_stnd)[0]
    
    ws, tl, testl = GD(y_tr_stnd, X_tr_stnd, y_test_stnd, X_test_stnd, ws_unif, pw_mdist_mat, 1e-5, 1e3, 10, 5, 'pw')
    ws_n = np.abs(ws)/np.sum(np.abs(ws))
    indices = np.flip(np.argsort(ws_n))
    
#    for i in np.flip(indices):
#        print(model_names[i])
    
    # ws_ns, tl_ns, testl_ns = SGD(y_tr[:100], X_tr[:,:100], y_test[:100], X_test[:,:100], ws_unif, hist_mtas, 1e-16, 0, 0, 100)
# =============================================================================
#     weights = np.array([.2, .2, .6])
#     dist_mat = np.array([[0,1,7],[1,0,7],[7,7,0]])
#     get_pointwise_loss_grad(weights, dist_mat)
#     x = np.ones(3)
#     Y = np.ones([3,3])
#     ws = x/3
#     pw_dist_mat = np.ones([3,3]) - np.eye(3)
#     l = 1
#     print(get_OLS_grad(x, Y, ws))
#     print(update_pwl_grad(x, Y, ws, pw_dist_mat, l, l))
#     ws = np.zeros(30)
#     ws[1] = 1
#     get_stdev_loss(ws, hist_data_stnd)
#     
# =============================================================================
