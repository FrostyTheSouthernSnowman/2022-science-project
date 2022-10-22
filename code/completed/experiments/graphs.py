"""
Script used to generate graphs for presentation
"""

import plotly.express as px
import pandas as pd
from experiment_02 import func

pd.options.plotting.backend = 'plotly'


def plot_all_bin_search_scores_by_time():
    df1 = pd.read_csv("../pages/raw_data/binary_search_neural_network.csv")
    df2 = pd.read_csv("../pages/raw_data/binary_search_modified.csv")
    df3 = pd.read_csv("../pages/raw_data/binary_search_modified_bug_fix.csv")

    combined = pd.DataFrame(
        {'simple': df1["binary_search"].apply(lambda x: func(x)), 'modified': df2["binary_search"].apply(lambda x: func(x)), 'modified with bug fix': df3["binary_search"].apply(lambda x: func(x))})
    combined.drop(combined.tail(1).index, inplace=True)

    fig = combined.plot()
    print("saving")
    fig.write_image("all_neural_network_bin_search.png")


def plot_all_scores_by_time():
    df1 = pd.read_csv("../pages/raw_data/binary_search_neural_network.csv")
    df2 = pd.read_csv("../pages/raw_data/binary_search_modified.csv")
    df3 = pd.read_csv("../pages/raw_data/binary_search_modified_bug_fix.csv")

    combined = pd.DataFrame(
        {'simple': df1["binary_search"].apply(lambda x: func(x)), 'modified': df2["binary_search"].apply(lambda x: func(x)), 'modified with bug fix': df3["binary_search"].apply(lambda x: func(x)), 'grad_descent': df3["grad_descent"].apply(lambda x: func(x))})
    combined.drop(combined.tail(1).index, inplace=True)

    fig = combined.plot()
    print("saving")
    fig.write_image("all_algos_neural_network.png")


def plot_func():
    x_vals = []
    incr = 0.125
    start = -27
    end = -7
    num = start
    while num <= end:
        x_vals.append(num)
        num += incr

    y_vals = [func(x) for x in x_vals]

    plt = px.line(x=x_vals, y=y_vals)
    plt.write_image("simple_neural_network.png")


plot_func()
