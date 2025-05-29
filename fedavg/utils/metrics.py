import numpy as np
import sklearn.metrics as sk


def cal_metric(labels, scores):

    fpr_list, tpr_list, thresholds = sk.roc_curve(labels, scores)
    fpr = fpr_list[np.argmax(tpr_list >= 0.95)]

    auroc = sk.auc(fpr_list, tpr_list)

    precision, recall, _ = sk.precision_recall_curve(labels, scores)
    aupr = sk.auc(recall, precision)

    return auroc, aupr, fpr
