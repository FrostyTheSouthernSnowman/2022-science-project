from functools import cache
import math
import numpy as np
import run

np.random.seed(21)

lr = 0.1
n_epochs = 20_000
number = -20


@cache
def sigmoid(n):
    return 1 / (1 + math.exp(-n))


@cache
def func(n):
    return (sigmoid(n*2+34) - 0.5)**2


@cache
def d_func(n):
    return (2*(math.exp(2*n+34)-1)*math.exp(2*n+34))/(math.exp(2*n+34)+1)**3


def min_grad_descent(lr, n_epochs, return_hist=False):
    global number
    num = number

    if return_hist:
        hist = []

    for _ in range(n_epochs):
        num -= d_func(num) * lr

        if return_hist:
            hist.append(num)

    if return_hist:
        return hist

    return num


def min_bin_search(n_epochs, lr, return_hist=True, *args, **kwargs):
    global number
    num = number

    if return_hist:
        hist = []

    num_prev = num
    grad = d_func(num) * lr
    num -= grad
    iters = 0
    for _ in range(n_epochs - iters):
        if func(num) > func(num_prev):
            break
        else:
            num_prev = num
            num -= grad

        if return_hist:
            hist.append(num)

        iters += 1

    for _ in range(n_epochs - iters):
        half_point = num_prev + ((num - num_prev) / 2)
        first_quartile = num_prev + ((half_point - num_prev) / 2)
        if func(first_quartile) < func(half_point):
            num = half_point

        else:
            num_prev = half_point

        if return_hist:
            hist.append(num)

    if return_hist:
        return hist, num

    return num


if __name__ == "__main__":
    run.run(func, min_bin_search,
            min_grad_descent, lr, n_epochs)
