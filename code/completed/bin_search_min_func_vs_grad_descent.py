from glob import glob
import numpy as np

lr = 0.1
n_epochs = 100
number = np.random.randint(10, 100) * np.random.random()


def func(n):
    return 2.5*(n**2) + 6.3*n + 32.54


def d_func(n):
    return 5*n + 6.3


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

    if abs(func(grad_descent_min)) < abs(func(bin_search_min)):
        print("gradient descent is better")

    elif abs(func(bin_search_min)) < abs(func(grad_descent_min)):
        print("binary search is better")

    else:
        print("equal or bug/error")
        print(f"gradient descent minimum: {grad_descent_min}")
        print(f"bin_search minimum: {bin_search_min}")
