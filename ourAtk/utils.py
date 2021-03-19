import scipy.sparse as sp
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import sys

def row_normalize(mx):
    """Row-normalize sparse matrix    from SGC"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def load_embeddings(embeddings_path):
    '''load embedding from tensorflow version DGI'''
    with open(embeddings_path, 'rb') as f:
        if sys.version_info > (3, 0):
            features = pkl.load(f, encoding='latin1')
        else:
            features = pkl.load(f)
    return features

def get_intersection(a,b):
    return list(set(a).intersection(set(b)))
    
def rowNorm(adj):
    '''for LP closed form'''
    adj = adj + sp.eye(adj.shape[0])
    rowsum = adj.sum(1).A1
    adj_normalized = sp.coo_matrix(adj/rowsum)
    return adj_normalized

    
def preprocess_graph(adj):
    """sys-normalize sparse matrix    from SGC"""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)
    
def check_norm(adj):
    print("check norm:",np.sum(adj.A, axis=1,keepdims=False))
    
def accuracy(y_pred, y_truth):
    correct = np.sum(y_pred == y_truth)
    return correct / len(y_truth)
    
def BinaryLabelToPosNeg(label):
    # turn label from {0,1} to {-1,1}
    label[label == 0] = -1
    return label

def BinaryLabelTo01(label):
    # turn label from {-1,1} to {0,1}
    label[label == -1] = 0
    return label
    
    
