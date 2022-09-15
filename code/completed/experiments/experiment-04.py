from functools import cache
import math
import numpy as np

np.random.seed(21)

lr = 0.1
step = 1
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


def min_bin_search_fixed_step_size(step, n_epochs, return_hist=False):
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
        if func(first_quartile) < func(half_point):
            num = half_point

        else:
            num_prev = half_point

        if return_hist:
            hist.append(num)

    if return_hist:
        return hist, num_prev

    return num


if __name__ == "__main__":
    grad_descent_mins = min_grad_descent(lr, n_epochs, return_hist=True)
    bin_search_mins, prev = min_bin_search_fixed_step_size(
        step, n_epochs, return_hist=True)

    x_vals = [bin_search_mins[-1] - 10, bin_search_mins[-1] - 9, bin_search_mins[-1] - 8, bin_search_mins[-1] - 7, bin_search_mins[-1] - 6, bin_search_mins[-1] - 5, bin_search_mins[-1] - 4, bin_search_mins[-1] - 3, bin_search_mins[-1] - 2, bin_search_mins[-1] - 1, bin_search_mins[-1], bin_search_mins[-1] + 1, bin_search_mins[-1] + 2,
              bin_search_mins[-1] + 3, bin_search_mins[-1] + 4, bin_search_mins[-1] + 5, bin_search_mins[-1] + 6, bin_search_mins[-1] + 7, bin_search_mins[-1] + 8, bin_search_mins[-1] + 9, bin_search_mins[-1] + 10]
    y_vals = [func(x) for x in x_vals]
    z = [x * 100 for x in range(n_epochs//100)]

    grad_descent_mins = grad_descent_mins[::100]
    bin_search_mins = bin_search_mins[::100]

    from plotly import graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=x_vals*(n_epochs//1000), y=[((x//len(x_vals)))*1000 for x in range((n_epochs//1000)*len(x_vals))], z=y_vals*(n_epochs//1000),
                  name="experiment-02: binary search is better. f(x) = sigmoid(x+34)"))  # ! THAT IS NOT F(X), FIX NOW
    fig.add_trace(go.Scatter3d(x=grad_descent_mins, y=z, z=[
                  func(x) for x in grad_descent_mins], mode="lines+markers", name=f"gradient descent minimum: {grad_descent_mins[-1]}"))
    fig.add_trace(go.Scatter3d(x=bin_search_mins, y=z, z=[
                  func(x) for x in bin_search_mins], mode="lines+markers", name=f"binary search minimum: {bin_search_mins[-1]}"))
    fig.show()

    if abs(func(grad_descent_mins[-1])) < abs(func(bin_search_mins[-1])):
        print("gradient descent is better")

    elif abs(func(bin_search_mins[-1])) < abs(func(grad_descent_mins[-1])):
        print("binary search is better")

    else:
        print("equal or bug/error")
        print(f"gradient descent minimum: {grad_descent_mins[-1]}")
        print(f"bin_search minimum: {bin_search_mins[-1]}")

    print(prev)
