import numpy as np

def RSE(pred, true, data_rse):
    return np.sqrt(np.mean((true - pred) ** 2)) / data_rse

def RAE(pred, true):
    return np.sum(np.abs(true - pred)) / np.sum(np.abs(true - true.mean()))

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def CORR(pred, true):
    sigma_p = pred.std(axis=0)
    sigma_g = true.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = true.mean(axis=0)
    index = (sigma_p * sigma_g != 0)
    correlation = ((pred - mean_p) * (true - mean_g)).mean(axis=0) / (sigma_p * sigma_g) 
    correlation = (correlation[index]).mean()
    return correlation




