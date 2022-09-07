import numpy as np

np.random.seed(77)

lr = 0.1
n_epochs = 100
number = np.random.randint(10, 100) * np.random.random()


def func(n):
    return n**2


def d_func(n):
    return 2*n


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


def min_bin_search(lr, n_epochs, return_hist=False):
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
        return hist

    return num


if __name__ == "__main__":
    grad_descent_mins = min_grad_descent(lr, n_epochs, return_hist=True)
    bin_search_mins = min_bin_search(lr, n_epochs, return_hist=True)

    x_vals = [bin_search_mins[-1] - 10, bin_search_mins[-1] - 9, bin_search_mins[-1] - 8, bin_search_mins[-1] - 7, bin_search_mins[-1] - 6, bin_search_mins[-1] - 5, bin_search_mins[-1] - 4, bin_search_mins[-1] - 3, bin_search_mins[-1] - 2, bin_search_mins[-1] - 1, bin_search_mins[-1], bin_search_mins[-1] + 1, bin_search_mins[-1] + 2,
              bin_search_mins[-1] + 3, bin_search_mins[-1] + 4, bin_search_mins[-1] + 5, bin_search_mins[-1] + 6, bin_search_mins[-1] + 7, bin_search_mins[-1] + 8, bin_search_mins[-1] + 9, bin_search_mins[-1] + 10]
    y_vals = [func(x) for x in x_vals]
    z = [x for x in range(n_epochs)]

    from plotly import graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=x_vals*(n_epochs), y=[((x//len(x_vals))) for x in range((n_epochs)*len(x_vals))], z=y_vals*(n_epochs),
                  name="experiment-01: binary search is better. f(x)=x**2"))
    fig.add_trace(go.Scatter3d(x=grad_descent_mins, y=z, z=[
                  func(x) for x in grad_descent_mins], mode="lines+markers", name=f"gradient descent minimum: {grad_descent_mins[-1]}"))
    fig.add_trace(go.Scatter3d(x=bin_search_mins, y=z, z=[
                  func(x) for x in bin_search_mins], mode="lines+markers", name=f"binary search minimum: {bin_search_mins[-1]}"))
    fig.show()
    fig.write_html("quadratic.md")

    if abs(func(grad_descent_mins[-1])) < abs(func(bin_search_mins[-1])):
        print("gradient descent is better")

    elif abs(func(bin_search_mins[-1])) < abs(func(grad_descent_mins[-1])):
        print("binary search is better")

    else:
        print("equal or bug/error")
        print(f"gradient descent minimum: {grad_descent_mins[-1]}")
        print(f"bin_search minimum: {bin_search_mins[-1]}")

    print(len(bin_search_mins))
    print(len(grad_descent_mins))
