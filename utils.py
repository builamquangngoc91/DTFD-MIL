import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def roc_threshold(label, prediction):
    """
    Fixed roc_threshold function to handle both binary and multi-class cases
    with proper handling of probability inputs
    """
    # Convert to numpy arrays if they aren't already
    if hasattr(label, 'cpu'):
        label = label.cpu().numpy()
    if hasattr(prediction, 'cpu'):
        prediction = prediction.cpu().numpy()
    
    label = np.array(label)
    prediction = np.array(prediction)
    
    # Get number of unique classes
    unique_classes = np.unique(label)
    n_classes = len(unique_classes)
    
    print(f"roc_threshold debug: n_classes={n_classes}, label_shape={label.shape}, pred_shape={prediction.shape}")
    print(f"label range: {label.min()}-{label.max()}, pred range: {prediction.min():.4f}-{prediction.max():.4f}")
    
    if n_classes == 2:
        # Binary classification
        if prediction.ndim > 1 and prediction.shape[1] > 1:
            # If prediction is probability matrix, take positive class probability
            prediction = prediction[:, 1]
        
        fpr, tpr, threshold = roc_curve(label, prediction)
        c_auc = roc_auc_score(label, prediction)
        
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        
        return c_auc, optimal_threshold
    
    else:
        # Multi-class classification
        if prediction.ndim == 1:
            # IMPORTANT: If prediction is 1D and contains probabilities (not class indices),
            # we need to handle this differently
            
            # Check if predictions look like probabilities vs class indices
            if prediction.dtype == np.float32 or prediction.dtype == np.float64:
                # These are probability scores, not class indices
                # For multi-class, we need probability matrix, but we only have 1D probabilities
                # This suggests the model output might be problematic
                
                # Convert 1D probabilities to class predictions for AUC calculation
                # Assuming equal probability threshold for all classes
                class_predictions = np.digitize(prediction, 
                                              bins=np.linspace(prediction.min(), prediction.max(), n_classes+1)[1:-1])
                
                # Create one-hot encoding from class predictions
                prediction_proba = np.eye(n_classes)[class_predictions]
                
            else:
                # These are class indices
                class_indices = prediction.astype(int)
                prediction_proba = np.eye(n_classes)[class_indices]
                
        else:
            # If prediction is 2D, assume it's already probability matrix
            prediction_proba = prediction
        
        # Binarize labels for multi-class AUC
        label_binarized = label_binarize(label, classes=unique_classes)
        
        # Handle case where there's only one class in the batch
        if label_binarized.shape[1] == 1:
            return 0.5, 0.5
        
        # Calculate multi-class AUC using 'ovr' (one-vs-rest) strategy
        try:
            c_auc = roc_auc_score(label_binarized, prediction_proba, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"AUC calculation failed: {e}")
            c_auc = 0.5
        
        # For multi-class, threshold doesn't have the same meaning
        optimal_threshold = 0.5
        
        return c_auc, optimal_threshold


def eval_metric(Y_pred, Y_true):
    """
    Fixed eval_metric function to handle both binary and multi-class cases
    with proper handling of probability inputs
    """
    # Convert to numpy arrays
    if hasattr(Y_pred, 'cpu'):
        Y_pred = Y_pred.cpu().numpy()
    if hasattr(Y_true, 'cpu'):
        Y_true = Y_true.cpu().numpy()
    
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)
    
    # Get number of unique classes
    unique_classes = np.unique(Y_true)
    n_classes = len(unique_classes)
    
    print(f"eval_metric debug: n_classes={n_classes}, Y_pred_shape={Y_pred.shape}, Y_true_shape={Y_true.shape}")
    print(f"Y_true range: {Y_true.min()}-{Y_true.max()}, Y_pred range: {Y_pred.min():.4f}-{Y_pred.max():.4f}")
    
    if n_classes == 2:
        # Binary classification
        if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
            # If Y_pred is probability matrix, convert to binary predictions
            Y_pred_binary = np.argmax(Y_pred, axis=1)
            Y_pred_proba = Y_pred[:, 1]  # Probability of positive class
        else:
            # If Y_pred is 1D, assume it's probabilities and convert to binary predictions
            Y_pred_binary = (Y_pred > 0.5).astype(int)
            Y_pred_proba = Y_pred
        
        # Calculate metrics
        acc = accuracy_score(Y_true, Y_pred_binary)
        precision = precision_score(Y_true, Y_pred_binary, zero_division=0)
        recall = recall_score(Y_true, Y_pred_binary, zero_division=0)
        
        # Specificity for binary classification
        tn = np.sum((Y_true == 0) & (Y_pred_binary == 0))
        fp = np.sum((Y_true == 0) & (Y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1 = f1_score(Y_true, Y_pred_binary, zero_division=0)
        auc, _ = roc_threshold(Y_true, Y_pred_proba)
        
        return acc, precision, recall, specificity, f1, auc
    
    else:
        # Multi-class classification
        if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
            # If Y_pred is probability matrix, convert to class predictions
            Y_pred_class = np.argmax(Y_pred, axis=1)
            Y_pred_proba = Y_pred
        else:
            # If Y_pred is 1D with probabilities, we need to convert to class predictions
            if Y_pred.dtype == np.float32 or Y_pred.dtype == np.float64:
                # Convert probabilities to class predictions using quantile-based binning
                Y_pred_class = np.digitize(Y_pred, 
                                         bins=np.linspace(Y_pred.min(), Y_pred.max(), n_classes+1)[1:-1])
                # Create probability matrix (this is an approximation)
                Y_pred_proba = np.eye(n_classes)[Y_pred_class]
            else:
                # Assume they're already class predictions
                Y_pred_class = Y_pred.astype(int)
                Y_pred_proba = np.eye(n_classes)[Y_pred_class]
        
        # Calculate metrics with multi-class averaging
        acc = accuracy_score(Y_true, Y_pred_class)
        precision = precision_score(Y_true, Y_pred_class, average='macro', zero_division=0)
        recall = recall_score(Y_true, Y_pred_class, average='macro', zero_division=0)
        f1 = f1_score(Y_true, Y_pred_class, average='macro', zero_division=0)
        
        # For multi-class, specificity is calculated per class and averaged
        specificities = []
        for class_idx in range(n_classes):
            # Convert to binary problem for each class
            y_true_binary = (Y_true == class_idx).astype(int)
            y_pred_binary = (Y_pred_class == class_idx).astype(int)
            
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            class_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(class_specificity)
        
        specificity = np.mean(specificities)
        
        # Calculate AUC
        auc, _ = roc_threshold(Y_true, Y_pred_proba)
        
        return acc, precision, recall, specificity, f1, auc


# Alternative: Quick fix that addresses the immediate IndexError
def quick_fix_eval_metric(Y_pred, Y_true):
    """
    Quick fix version that handles the immediate problem
    """
    # Convert to numpy arrays
    if hasattr(Y_pred, 'cpu'):
        Y_pred = Y_pred.cpu().numpy()
    if hasattr(Y_true, 'cpu'):
        Y_true = Y_true.cpu().numpy()
    
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)
    
    n_classes = len(np.unique(Y_true))
    
    if n_classes == 2:
        # Binary case - original logic
        if Y_pred.ndim > 1:
            oprob = Y_pred[:, 1]
        else:
            oprob = Y_pred
        
        label = Y_true
        auc, threshold = roc_threshold(label, oprob)
        
        # Convert probabilities to predictions for other metrics
        pred_binary = (oprob > threshold).astype(int)
        acc = accuracy_score(Y_true, pred_binary)
        precision = precision_score(Y_true, pred_binary, zero_division=0)
        recall = recall_score(Y_true, pred_binary, zero_division=0)
        f1 = f1_score(Y_true, pred_binary, zero_division=0)
        
        # Specificity
        tn = np.sum((Y_true == 0) & (pred_binary == 0))
        fp = np.sum((Y_true == 0) & (pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return acc, precision, recall, specificity, f1, auc
    
    else:
        # Multi-class case
        if Y_pred.ndim > 1:
            # Probability matrix
            pred_class = np.argmax(Y_pred, axis=1)
            pred_proba = Y_pred
        else:
            # 1D probabilities - convert to class predictions
            # Use quantile-based binning
            thresholds = np.quantile(Y_pred, np.linspace(0, 1, n_classes+1))
            pred_class = np.digitize(Y_pred, thresholds[1:-1])
            pred_proba = np.eye(n_classes)[pred_class]
        
        # Calculate metrics
        acc = accuracy_score(Y_true, pred_class)
        precision = precision_score(Y_true, pred_class, average='macro', zero_division=0)
        recall = recall_score(Y_true, pred_class, average='macro', zero_division=0)
        f1 = f1_score(Y_true, pred_class, average='macro', zero_division=0)
        
        # Specificity (macro-averaged)
        specificities = []
        for i in range(n_classes):
            y_true_bin = (Y_true == i).astype(int)
            y_pred_bin = (pred_class == i).astype(int)
            tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
            fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        specificity = np.mean(specificities)
        
        # AUC using one-vs-rest
        try:
            label_bin = label_binarize(Y_true, classes=range(n_classes))
            if label_bin.shape[1] == 1:
                auc = 0.5
            else:
                auc = roc_auc_score(label_bin, pred_proba, multi_class='ovr', average='macro')
        except:
            auc = 0.5
        
        return acc, precision, recall, specificity, f1, auc