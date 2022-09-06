from functools import cache
import math
import numpy as np

np.random.seed(21)

lr = 0.1
n_epochs = 100
number = np.random.randint(-50, 50) * np.random.random()

number = -20


@cache
def sigmoid(n):
    return 1 / (1 + math.exp(-n))


@cache
def func(n):
    return sigmoid(n*2+34)


@cache
def d_func(n):
    return -((2*(math.exp(n*2+34)))/((math.exp(n*2+34) + 1)**2))


def min_grad_descent(lr, n_epochs):
    global number
    num = number

    for _ in range(n_epochs):
        num -= d_func(num) * lr

    return num


def min_bin_search(lr, n_epochs):
    global number
    num = number

    num_prev = num
    grad = d_func(num) * lr
    num -= grad
    iters = 1
    for _ in range(n_epochs - iters):
        if func(num) > func(num_prev):
            break
        else:
            num_prev = num
            num -= grad

        iters += 1

    for _ in range(n_epochs - iters):
        half_point = num_prev + ((num - num_prev) / 2)
        first_quartile = num_prev + ((half_point - num_prev) / 2)
        if func(first_quartile) < func(half_point):
            num = half_point

        else:
            num_prev = half_point

    return num


if __name__ == "__main__":
    grad_descent_min = min_grad_descent(lr, n_epochs)
    bin_search_min = min_bin_search(lr, n_epochs)

    x_vals = [bin_search_min - 10, bin_search_min - 9, bin_search_min - 8, bin_search_min - 7, bin_search_min - 6, bin_search_min - 5, bin_search_min - 4, bin_search_min - 3, bin_search_min - 2, bin_search_min - 1, bin_search_min, bin_search_min + 1, bin_search_min + 2,
              bin_search_min + 3, bin_search_min + 4, bin_search_min + 5, bin_search_min + 6, bin_search_min + 7, bin_search_min + 8, bin_search_min + 9, bin_search_min + 10]
    y_vals = [func(x) for x in x_vals]

    from plotly import graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="func"))
    fig.add_trace(go.Scatter(x=[grad_descent_min], y=[
                  func(grad_descent_min)], mode="markers", name=f"gradient descent minimum: {grad_descent_min}"))
    fig.add_trace(go.Scatter(x=[bin_search_min], y=[
                  func(bin_search_min)], mode="markers", name=f"binary search minimum: {bin_search_min}"))
    fig.show()

    if abs(func(grad_descent_min)) < abs(func(bin_search_min)):
        print("gradient descent is better")

    elif abs(func(bin_search_min)) < abs(func(grad_descent_min)):
        print("binary search is better")

    else:
        print("equal or bug/error")
        print(f"gradient descent minimum: {grad_descent_min}")
        print(f"bin_search minimum: {bin_search_min}")
