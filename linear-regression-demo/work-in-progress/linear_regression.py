import math
from typing import Tuple, Union
import plotly.express as px
import numpy as np
from enum import Enum


class Algorithms(Enum):
    GRADIENT_DESCENT = 1
    GRADIENT_APPROXAMATOR = 2


x = np.array([0, 1, 2]).reshape(-1, 1)
y = np.array([1, 3, 5]).reshape(-1, 1)


class Linear():
    def __init__(self, lr: int):
        """
        Simplest MultiDimBinS implementation
        """

        self.w = np.random.random()
        self.b = np.random.random()
        self.lr = lr
        self.x_shape = None
        self.w_grad_hist = []
        self.b_grad_hist = []

    def fit(self, X: np.array, Y: np.array, n_epochs: int, algo: Algorithms, include_stats: bool = False) -> Union[np.array, float]:
        """
        trains the linear model
        params:
            X: np.array containing the x values
            Y: np.array containing the y values
            n_epochs: int, the number of epochs to loop through
            algo: Algorithms, the algorithm used to fit the models
            include_stats: bool, a flag for including statistics

        returns:
            np.array: weight matrix
            float: b also known as the intercept in the linear equation mx+b=y
        """
        self.loss_hist = []

        self.x_shape = X.shape

        if algo is Algorithms.GRADIENT_DESCENT:
            for _ in range(n_epochs):
                loss, _, _ = self.fit_gradient_descent_epoch(X, Y)
                if include_stats:
                    self.loss_hist.append(loss)

        if algo is Algorithms.GRADIENT_APPROXAMATOR:
            if self.lr == 1:
                self.lr = 0.9999999999

            prev_loss = self.get_loss(X, Y)
            loss, w_grad_approx, b_grad_approx = self.fit_gradient_descent_epoch(
                X, Y)
            print(w_grad_approx)
            if include_stats:
                self.loss_hist.append(loss)
                self.loss_hist.append(prev_loss)

            for i in range(n_epochs - 1):
                prev_loss = loss
                loss = self.get_loss(X, Y)
                w_grad_approx = w_grad_approx * self.lr * \
                    (prev_loss-loss)
                b_grad_approx = b_grad_approx * self.lr * \
                    (prev_loss-loss)

                self.w_grad_hist.append(w_grad_approx)
                self.b_grad_hist.append(b_grad_approx)

                self.w -= w_grad_approx
                self.b -= b_grad_approx
                print(loss-prev_loss)
                if include_stats:
                    self.loss_hist.append(prev_loss)

        return self.w, self.b

    def fit_gradient_descent_epoch(self, X: np.array, Y: np.array) -> Tuple[float, float, float]:
        """
        fits a single epoch

        params:
            x: np.array containing the x values
            y: np.array containing the y values

        returns:
            float: error rate calculated with the mean squared error
            float: w gradient
            float: b gradient
        """

        d_cost_over_d_w, d_cost_over_d_b = self.gradient(X, Y, self.x_shape)

        error = self.get_loss(X, Y)

        # updating the weights with the calculated gradients
        self.w -= self.lr*d_cost_over_d_w
        # updating the weights with the calculated gradients
        self.b -= self.lr*d_cost_over_d_b

        return error, d_cost_over_d_w, d_cost_over_d_b

    def gradient(self, X: np.array, Y: np.array, x_shape) -> Tuple[float, float]:
        """
        Function to get the gradient for the first step

        params:
            x: np.array containing the x values
            y: np.array containing the y values

        returns:
            float: w gradient
            float: b gradient
        """

        # partial derivative of cost w.r.t self.w
        d_cost_over_d_w = (-2/x_shape[0]) * np.sum(X *
                                                   (Y - self.predict(X)))

        # partial derivative of cost w.r.t self.b
        d_cost_over_d_b = (-2/x_shape[0]) * \
            np.sum(Y - self.predict(X))

        return d_cost_over_d_w, d_cost_over_d_b

    def get_loss(self, X: np.array, Y: np.array) -> float:
        """
        gets the current loss

        params:
            x: np.array containing the x values
            y: np.array containing the y values

        returns:
            float: the loss
        """

        return np.square(np.subtract(self.predict(X), Y)).mean()

    def predict(self, X) -> float:
        return self.w * X + self.b


model = Linear(0.1)

print(model.fit(x, y, 20, Algorithms.GRADIENT_APPROXAMATOR, include_stats=True))
fig = px.line(y=model.loss_hist)
fig.show()
