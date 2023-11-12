import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from characteristics import mean_vals, st_deviation

def nan_check(df):
    missed = df.isna().sum().sum()
    if missed > 0:
        print("Количество пропущенных значений: ")
        for col in df.columns:
            if type(df[col].to_dict()[0]) == int or type(df[col].to_dict()[0]) == float:
                df[col].fillna(df[col].mean(), inplace = True)
            else:
                df[col].fillna(df[col].mode().iloc[0], inplace = True)
    else:
        print("Пропущенных значений нет")

def cat_features (df):
    flag = 0
    for col in df.columns:
        if type(df[col].to_dict()[0]) == str:
            flag = 1
            cat_col = pd.get_dummies(df[col],  drop_first=True, dtype=int)
            to_add = pd.DataFrame(data = cat_col.values, columns=[df[col].name])
            df = df.drop(df[col].name, axis = 1)
            df_good = pd.concat([df, to_add], axis = 1)


    if flag == 0:
        print("Категориальных признаков нет")
        return df
    else:
        print("Категориальные признаки изменены")
        return df_good

def define_distrib(df):
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(3, 3, i + 1)
        df[col].plot(kind='hist')
        plt.title(df[col].name)
    plt.tight_layout()
    plt.show()

def std_scaler(df):
    cnt = 0
    shape = df.shape
    means = mean_vals(df, shape[0])
    devs = st_deviation(df, means, shape[0])
    #используется для нормального и геометрического распределения
    for col in df.columns:
        if df[col].name != 'Outcome':
            df.loc[:, col] = (df[col] - means[cnt]) / devs[cnt]
        cnt += 1

def min_max_scaler(df):
    #используется для равномерного распределения
    min_val = df.min()
    max_val = df.max()
    df = (df - min_val)/(max_val - min_val)


def splitter (X, y, test_size, random_state):
    rows = X.shape[0]
    ids = np.array(range(rows))
    random.seed(random_state)
    random.shuffle(ids)

    test = round(rows * test_size)

    test_ids = ids[0:test]
    tr_ids = ids[test:rows]

    X_train = pd.DataFrame(X.values[tr_ids, :], columns=X.columns)
    X_test = pd.DataFrame(X.values[test_ids, :], columns=X.columns)
    y_train = pd.DataFrame(y.values[tr_ids], columns=['Outcome'])
    y_test = pd.DataFrame(y.values[test_ids], columns=['Outcome'])
    return X_train, X_test, y_train, y_test


def prep_data(X, y):
    min_max_scaler(y)
    std_scaler(X)

    X_train, X_test, y_train, y_test = splitter(X, y, 0.15, 41)
    return X_train, X_test, y_train, y_test
