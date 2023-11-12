from preps import *
from log_regression import *
from characteristics import *

df = pd.read_csv('diabetes.csv')
define_distrib(df)
show_corr_matrix(df)
nan_check(df)
cat_features(df)
X = df.drop(['Pregnancies', 'Outcome'], axis = 1)
y = df['Outcome']
X_train, X_test, y_train, y_test = prep_data(X, y)
d1_y_train = y_train.values.reshape(1, y_train.shape[0])[0]

iters = [10, 100, 500, 1000, 5000, 10000]
rates = [0.1, 0.01, 0.001]
methods = ['newton', 'grad_dec']
acc_cur = -1
prec_cur = -1
recall_cur = -1
f1_cur = -1
best_iter = 0
best_rate = 0
acc_best = -1
prec_best = -1
recall_best = -1
f1_best = -1
method_best = ''
for method in methods:
    for it in iters:
        for r in rates:
            lr_t = Log_Reg(max_iter=it, learning_rate=r, method=method)
            lr_t.fit(X_train.values, d1_y_train)
            y_pr = lr_t.predict(X_test.values)
            acc, precision, recall, f1 = metrics(y_test.values, y_pr)
            print(f'iterations: {it}, learning_rate: {r}, method: {method}')
            print(f'Accuracy: {acc} \nPrecision: {precision} \nRecall: {recall} \nF1_score: {f1}\n')
            if f1 >= f1_cur and (acc >= acc_cur or precision >= acc_cur or recall >= recall_cur):
                best_rate = r
                best_iter = it
                method_best = method
                acc_best, prec_best, recall_best, f1_best = acc, precision, recall, f1
                acc_cur, prec_cur, recall_cur, f1_cur = acc, precision, recall, f1


print(f'Наилучший результат: iterations: {best_iter}, learning_rate: {best_rate}, method: {method_best}\n')

print("Построенная модель")
lr = Log_Reg(max_iter=best_iter, learning_rate=best_rate, method='grad_dec')

lr.fit(X_train.values, d1_y_train)
y_pred = lr.predict(X_test.values)

acc, precision, recall, f1 = metrics(y_test.values, y_pred)
print(f'Accuracy: {acc} \nPrecision: {precision} \nRecall: {recall} \nF1_score: {f1}')

print("Потери: ", loss_func(lr.get_probs(X_test.values), y_test.values)[0])


