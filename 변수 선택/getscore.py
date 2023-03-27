
def get_eval_score(y_test, y_pred, y_pred_proba):
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix,accuracy_score, precision_score, recall_score
    confusion = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("-"* 100)
    print("오차행렬:")
    print(confusion)
    print("정확도: {:.7f} 정밀도: {:.7f} 재현율: {:.7f} F1:{:.7f} AUC: {:.7f}".\
          format(acc, prec,recall, f1, roc_auc ))
    return
