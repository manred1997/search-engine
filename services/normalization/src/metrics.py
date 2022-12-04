import numpy as np
from sklearn.metrics import f1_score, classification_report


def compute_metrics(cap_preds, cap_labels, punc_preds, punc_labels, label_masks, loss_coef):
    punc_label_set = [ 'O', 'PERIOD', 'COMMA', 'QMARK','[CLS]', '[SEP]']
    punc_label_map = {label: i for i, label in enumerate(punc_label_set)}

    cap_preds_list, cap_labels_list, punc_preds_list, punc_labels_list = [], [], [], []

    for i in range(cap_preds.shape[0]):
        for j in range(cap_preds.shape[1]):
            if label_masks[i, j] != 0:
                cap_labels_list.append(cap_labels[i][j])
                cap_preds_list.append(cap_preds[i][j])
                punc_labels_list.append(punc_label_set[punc_labels[i][j]])
                punc_preds_list.append(punc_label_set[punc_preds[i][j]])
    
    f1_cap = f1_score(cap_labels_list,cap_preds_list, labels=[1,2],average='micro')
    f1_punc = f1_score(punc_labels_list,punc_preds_list,labels=['PERIOD', 'COMMA', 'QMARK'],average='micro')
    cap_report = 0

    cap_report = classification_report(cap_labels_list,cap_preds_list, labels=[1,2],digits=4)
    punc_report = classification_report(punc_labels_list,punc_preds_list,labels=['PERIOD', 'COMMA', 'QMARK'],digits=4)
    
    

    results = {
        'f1_cap': f1_cap,
        'f1_punc': f1_punc,
        'cap_report': cap_report,
        'punc_report': punc_report,
        'mean_f1': ((10*loss_coef)*f1_cap+(10*(1-loss_coef))*f1_punc)/10
    }
    return results
