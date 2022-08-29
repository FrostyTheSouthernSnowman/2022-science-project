import numpy as np

lr = 0.1
n_epochs = 100
number = np.random.randint(10, 100) * np.random.random()


def func(n):
    return 2.5*(n**2) + 6.3*n + 32.54


def d_func(n):
    return 5*n + 6.3


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


bin_search_min = min_bin_search(lr, n_epochs)
print(f"minimum value input: {bin_search_min}")
print(f"minimum value of function: {func(bin_search_min)}")

x_vals = [bin_search_min - 10, bin_search_min - 9, bin_search_min - 8, bin_search_min - 7, bin_search_min - 6, bin_search_min - 5, bin_search_min - 4, bin_search_min - 3, bin_search_min - 2, bin_search_min - 1, bin_search_min, bin_search_min + 1, bin_search_min + 2,
          bin_search_min + 3, bin_search_min + 4, bin_search_min + 5, bin_search_min + 6, bin_search_min + 7, bin_search_min + 8, bin_search_min + 9, bin_search_min + 10]
y_vals = [func(x) for x in x_vals]

"""
Optional graphing code

from plotly import graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_vals))
fig.add_trace(go.Scatter(x=[bin_search_min], y=[func(bin_search_min)]))
fig.show()

"""
