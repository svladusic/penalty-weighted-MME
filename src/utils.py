import numpy as np
from scipy.optimize import nnls

def get_pointwise_loss(ws: np.array, pw_dist_mat: np.array) -> float:
    """Get loss for point-wise regularization function. Adding this to non-negative least squares loss yields total loss for gradient descent.

    Args:
        ws (np.array): vector of fitted weights/parameter estimates in non-negative regression model.
        pw_dist_mat (np.array): point-wise distances between models in latent space.

    Returns:
        loss (float): total loss for point-wise regulatization function.
    """
    W = np.outer(ws,ws)
    w_dist_mat = W*pw_dist_mat
    loss = np.sum(w_dist_mat)/2
    return loss


def get_pointwise_loss_grad(ws: np.array, pw_dist_mat:np.array) -> np.array:
    """Get gradient for point-wise loss function given point-wise distances and nnls weights. Needed to perform gradient descent.

    Args:
        ws (np.array): vector of fitted weights/parameter estimates in non-negative regression model.
        pw_dist_mat (np.array): point-wise distances between models in latent space.

    Returns:
        loss_grad (np.array): vector of loss gradient components from point-wise regulatization function. 
    """
    n = np.shape(pw_dist_mat)[0]
    loss_grad = np.zeros(n)
    for i in range(n):
        loss_grad[i] = np.sum(np.abs(np.delete(ws, i))/np.delete(pw_dist_mat[i,:], i))  
    return loss_grad


def get_stdev_loss(ws:np.array, dist_mat:np.array) -> float:
    """Get loss for standard deviation regularization function. Adding output to non-negative least squares loss yields total loss for gradient descent.

    Args:
        ws (np.array): vector of fitted weights/parameter estimates in non-negative regression model.
        dist_mat (np.array): distances between models in latent space.

    Returns:
        loss (float): total loss for standard deviation regulatization function.
    """
    w_norm = np.sum(ws)
    mu = np.mean(dist_mat[-1], axis=0)
    dw = np.array([np.linalg.norm(d-mu) for d in dist_mat])
    var_num = (np.sum(np.abs(ws)*dw))
    loss = w_norm/var_num
    return loss


def get_stdev_loss_grad(ws:np.array, dist_mat:np.array) -> np.array:
    """Get gradient for point-wise loss function given point-wise distances and nnls weights. Needed to perform gradient descent.

    Args:
        ws (np.array): vector of fitted weights/parameter estimates in non-negative regression model.
        dist_mat (np.array): distances between models in latent space.

    Returns:
        loss_grad (np.array): vector of loss gradient components from standard deviation regulatization function. 
    """
    mu = np.mean(dist_mat[-1], axis=0)
    dw = np.array([np.linalg.norm(d-mu) for d in dist_mat])
    w_denom = (np.sum(np.abs(ws)*dw))
    w_num = np.sum(np.abs(ws))
    sws = np.sign(ws)
    loss_grad = sws*(1/w_denom - dw*w_num/(w_denom**2))
    return loss_grad


def get_family_loss(ws: np.array, families:list) -> float:
    """Get loss for standard deviation regularization function. Adding output to non-negative least squares loss yields total loss for gradient descent.

    Args:
        ws (np.array): vector of fitted weights/parameter estimates in non-negative regression model.
        families (list): list of lists. each internal list inside specifies whether or not a CMAP model is included in some model family.

    Returns:
        loss (float): total loss for standard deviation regulatization function.  
    """
    family_losses = [np.sum(ws[f])/np.sum(np.delete(ws, f)) for f in families]
    loss = sum(family_losses)
    return loss        


def get_family_loss_grad(ws: np.array, families: list) -> np.array:
    """Get gradient for point-wise loss function given family regularization function and nnls weights. Needed to perform gradient descent.

    Args:
        ws (np.array): vector of fitted weights/parameter estimates in non-negative regression model.
        families (list): list of lists. each internal list inside specifies whether or not a CMAP model is included in some model family.

    Returns:
        loss_grad (np.array): vector of loss gradient components from family regulatization function. 
    """
    #TODO: Refactor with list comprehension for ease of understanding.
    n = np.size(ws)
    loss_grad = np.zeros(n)
    for k in range(n):
        for f in families:
            num = np.sum(ws[f])
            denom = np.sum(np.delete(ws, f))**2
            if k in f:
                loss_grad[k] += 1
            else:
                loss_grad[k] += -num/denom
    loss_grad = loss_grad*np.sign(ws)
    return loss_grad


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


def full_loss_std(x, Y, ws, dist_mat, l1, l2):
    return get_OLS_loss(x, Y, ws) + l1*get_stdev_loss(ws, dist_mat) + \
           l2*np.linalg.norm(ws)**2


def update_std_grad(x, Y, ws, families, l1, l2):
    return get_OLS_grad(x, Y, ws) + \
           l1*get_family_loss_grad(ws, families) + l2*ws 

def full_loss_fam(x, Y, ws, families, l1, l2):
    return get_OLS_loss(x, Y, ws) + l1*get_family_loss(ws, families) + \
           l2*np.linalg.norm(ws)**2


def update_fam_grad(x, Y, ws, families, l1, l2):
    return get_OLS_grad(x, Y, ws) + \
           l1*get_family_loss_grad(ws, families) + l2*ws 
           
           
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
