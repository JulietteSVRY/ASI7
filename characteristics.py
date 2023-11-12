import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def mean_vals(df, rows): #среднее значение
    means = []
    for col in df.columns:
        means.append(df[col].sum() / rows)
    return means

def st_deviation(df, means, n): #стандартное отклонение насколько значения в наборе данных различаются от среднего значения.
    cnt = 0
    devs = []
    for col in df.columns:
        tmp = sum((x - means[cnt]) ** 2 for x in df[col].values) / n
        st_dev = np.sqrt(tmp)
        devs.append(st_dev)
        cnt += 1
    return devs

def conf_matrix(y_test, y_pred):
    cm = {
        'TP': 0,  # True Positives
        'TN': 0,  # True Negatives
        'FP': 0,  # False Positives
        'FN': 0   # False Negatives
    }
    for true, pred in zip(y_test, y_pred):
        if true == 1 and pred == 1:
            cm['TP'] += 1
        elif true == 0 and pred == 0:
            cm['TN'] += 1
        elif true == 0 and pred == 1:
            cm['FP'] += 1
        elif true == 1 and pred == 0:
            cm['FN'] += 1

    return cm

def metrics(y_test, y_prend):
    cm = conf_matrix(y_test, y_prend)
    acc = (cm['TP'] + cm['TN']) / (cm['TP'] + cm['TN'] + cm['FN'] + cm['FP'])
    precision = cm['TP'] / (cm['TP'] + cm['FP'])
    recall = cm['TP'] / (cm['TP'] + cm['FN'])
    f1 = 2 * (precision * recall) / (precision + recall)

    return acc, precision, recall, f1

def show_corr_matrix(X_train):
    corred = X_train.corr().round(2)
    sns.heatmap(corred, annot = True)
    plt.show()