from sklearn.metrics import roc_auc_score, roc_curve
import torch
import numpy as np
from sklearn.preprocessing import label_binarize

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def roc_threshold(label, prediction):
    if hasattr(label, 'cpu'):
        label = label.cpu().numpy()
    if hasattr(prediction, 'cpu'):
        prediction = prediction.cpu().numpy()
    
    label = np.array(label)
    prediction = np.array(prediction)
    
    # Check if binary or multi-class
    n_classes = len(np.unique(label))
    
    if n_classes == 2:
        # Binary classification
        if prediction.ndim > 1:
            prediction = prediction[:, 1]
        
        fpr, tpr, threshold = roc_curve(label, prediction)
        c_auc = roc_auc_score(label, prediction)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
    else:
        # Multi-class classification
        if prediction.ndim == 1:
            prediction = np.eye(n_classes)[prediction]
        
        label_bin = label_binarize(label, classes=range(n_classes))
        
        if label_bin.shape[1] == 1:
            c_auc = 0.5
        else:
            # KEY FIX: Add multi_class parameter
            c_auc = roc_auc_score(label_bin, prediction, multi_class='ovr', average='macro')
        
        optimal_threshold = 0.5
    
    return c_auc, optimal_threshold

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc
