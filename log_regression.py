import numpy as np

class Log_Reg():
    def __init__(self, max_iter = 1000, learning_rate = 0.01, method = "grad_dec"):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.method = method
        self.weights = None

    def sigmoid_func(self, z): #преобразует линейную комбинацию признаков в вероятность принадлежности к одному из классов
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y): #обучение с использованием градиентного спуска (`grad_dec`) или метода Ньютона (`newton_opt`) в зависимости от выбранного метода оптимизации
        X = self.addition_term(X)
        self.weights = np.zeros(X.shape[1])
        if self.method == 'grad_dec':
            self.grad_dec(X, y)
        else:
            self.newton_opt(X, y)

    def addition_term(self, X): #добавляет дополнительный признак (терм) к матрице признаков `X`. Этот терм является постоянным и равен 1, чтобы учесть свободный член в линейной регрессии
        term = np.ones((X.shape[0], 1))
        return np.concatenate((term, X), axis=1)

    def get_probs(self, X): #возвращает вероятности класса 1 (положительного класса) для входных признаков `X`.
        X = self.addition_term(X)
        z = np.matmul(X, self.weights)
        return self.sigmoid_func(z)

    def predict(self, X): #выполняет бинарное предсказание (0 или 1) на основе вероятностей, полученных с помощью `get_probs`
        X = self.addition_term(X)
        z = np.matmul(X, self.weights)

        return (self.sigmoid_func(z) > 0.5).astype(int)

    def grad_dec(self, X, y): #первая производная Ньютона, для обновления параметров модели с целью минимизации функции потерь
        for i in range(self.max_iter):
            z = np.matmul(X, self.weights)
            preds = self.sigmoid_func(z)
            #транспонируем матрицу хар-тик, каждое значение икса в строке умножаем на разность
            # прогноза и факта
            gradient = np.matmul(X.T, (preds - y)) / X.shape[1]
            prev_weights = self.weights
            self.weights = self.weights - self.learning_rate * gradient
            if np.sum(abs(self.weights - prev_weights)) < 1e-6:
                break

    def hessian_cnt(self, X): #вычисления Гессиана вторая производная (матрицы вторых производных) для метода оптимизации Ньютона
        z = np.matmul(X, self.weights)
        h = self.sigmoid_func(z)
        R = np.diag(h * (1 - h))
        return np.matmul(X.T, np.matmul(R, X))
    def newton_opt(self, X, y): # в ряд Тейлора и нахождении её локального минимума (или максимума)
        for i in range(self.max_iter):
            z = np.matmul(X, self.weights)
            preds = self.sigmoid_func(z)
            gradient = np.matmul(X.T, (preds - y)) / X.shape[1]
            hessian = self.hessian_cnt(X)
            self.weights = self.weights - self.learning_rate * np.linalg.inv(hessian) @ gradient
            prev_weights = self.weights
            self.weights -= np.linalg.inv(hessian) @ gradient

            if np.sum(abs(self.weights - prev_weights)) < 1e-6:
                break



def loss_func(probabilities, Y): #функция для вычисления функции потерь (логарифмической функции потерь) между вероятностями `probabilities` и фактическими метками `Y`
    tmp = 0
    epsilon = 1e-15  #небольшой эпсилон для логарифма
    for p, y in zip(probabilities, Y):
        p = np.maximum(epsilon, np.minimum(1 - epsilon, p))  #применяем ограничение к p
        tmp += (y * np.log(p) + (1 - y) * np.log(1 - p))
    log_loss = - (1 / len(Y)) * tmp
    return log_loss

#функция потерь и градиента графиуи, влияние каждого гиперпараметра
#то такое каждый метод ньютона и тд, как применяется
# оптимизатор adam
#Сделайте выводы о том, какие значения гиперпараметров наилучшим образом работают для данного набора данных и задачи классификации. Обратите внимание на изменение производительности модели при варьировании гиперпараметров.