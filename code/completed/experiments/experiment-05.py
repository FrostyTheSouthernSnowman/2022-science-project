from functools import cache
import math
import numpy as np
import run

np.random.seed(77)

lr = 0.1
step = 1
n_epochs = 20_000
number = np.random.randint(-30, 30) * np.random.random()
number = -13.2358147


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
        hist.append(num)

    for _ in range(n_epochs):
        num -= d_func(num) * lr

        if return_hist:
            hist.append(num)

    if return_hist:
        return hist

    return num


def min_bin_search_fixed_step_size_with_bug_fix(n_epochs, step, return_hist=True, *args, **kwargs):
    global number
    num = number

    if return_hist:
        hist = []
        hist.append(num)

    num_prev = num
    prev_num_prev = num
    grad = d_func(num) * lr
    num -= grad
    iters = 0
    for _ in range(n_epochs - iters):
        if func(num) > func(num_prev):
            num_prev = prev_num_prev
            break
        else:
            prev_num_prev = num_prev
            num_prev = num
            if abs(grad) < step:
                if grad < 0:
                    grad = -step
                if grad > 0:
                    grad = step
                if grad == 0:
                    break

            num -= grad

        if return_hist:
            hist.append(num)

        iters += 1

    for _ in range(n_epochs - iters):
        half_point = num_prev + ((num - num_prev) / 2)
        first_quartile = num_prev + ((half_point - num_prev) / 2)
        third_quartile = num + ((half_point - num) / 2)

        first_quartile_loss = func(first_quartile)
        half_point_loss = func(half_point)
        third_quartile_loss = func(third_quartile)

        if first_quartile_loss < half_point_loss:
            num = half_point

        elif third_quartile_loss < half_point_loss:
            num_prev = half_point

        else:
            num = third_quartile
            num_prev = first_quartile

        if return_hist:
            hist.append(num)

    if return_hist:
        return hist, num

    return num


if __name__ == "__main__":
    run.run(func, min_bin_search_fixed_step_size_with_bug_fix,
            min_grad_descent, lr, n_epochs, step)
