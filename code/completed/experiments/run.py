import argparse
import pandas as pd


def run(func, bin_search, grad_descent, lr, n_epochs, step=None, small=False):
    grad_descent_mins = grad_descent(lr, n_epochs, return_hist=True)
    bin_search_mins, min = bin_search(
        n_epochs=n_epochs, lr=lr, step=step, return_hist=True)

    x_vals = [bin_search_mins[-1] - 10, bin_search_mins[-1] - 9, bin_search_mins[-1] - 8, bin_search_mins[-1] - 7, bin_search_mins[-1] - 6, bin_search_mins[-1] - 5, bin_search_mins[-1] - 4, bin_search_mins[-1] - 3, bin_search_mins[-1] - 2, bin_search_mins[-1] - 1, bin_search_mins[-1], bin_search_mins[-1] + 1, bin_search_mins[-1] + 2,
              bin_search_mins[-1] + 3, bin_search_mins[-1] + 4, bin_search_mins[-1] + 5, bin_search_mins[-1] + 6, bin_search_mins[-1] + 7, bin_search_mins[-1] + 8, bin_search_mins[-1] + 9, bin_search_mins[-1] + 10]
    y_vals = [func(x) for x in x_vals]
    if small:
        z = [x for x in range(n_epochs)]

    else:
        z = [x * 25 for x in range(n_epochs//25)]

        grad_descent_mins = grad_descent_mins[::25]
        bin_search_mins = bin_search_mins[::25]

    from plotly import graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=x_vals*(n_epochs//1000), y=[((x//len(x_vals)))*1000 for x in range((n_epochs//1000)*len(x_vals))], z=y_vals*(n_epochs//1000),
                            name="experiment-02"))
    fig.add_trace(go.Scatter3d(x=grad_descent_mins, y=z, z=[
                  func(x) for x in grad_descent_mins], mode="lines+markers", name=f"gradient descent minimum: {grad_descent_mins[-1]}"))
    fig.add_trace(go.Scatter3d(x=bin_search_mins, y=z, z=[
                  func(x) for x in bin_search_mins], mode="lines+markers", name=f"binary search minimum: {bin_search_mins[-1]}"))

    if abs(func(grad_descent_mins[-1])) < abs(func(bin_search_mins[-1])):
        print("gradient descent is better")

    elif abs(func(bin_search_mins[-1])) < abs(func(grad_descent_mins[-1])):
        print("binary search is better")

    else:
        print("equal or bug/error")
        print(f"gradient descent minimum: {grad_descent_mins[-1]}")
        print(f"bin_search minimum: {bin_search_mins[-1]}")

    parser = argparse.ArgumentParser(
        description="run gradient descent and binary search side by side")
    parser.add_argument("--save", metavar="path",
                        help="Save the plot to some location")
    parser.add_argument("--type", metavar="type",
                        help="What type of data should be stored e.g. html, img, json, or raw_data")

    parser.add_argument("--show", help="should the plot be displayed",
                        action=argparse.BooleanOptionalAction)

    args = vars(parser.parse_args())

    if args['save'] != None:
        if args['type'] == "raw_data":
            # save csv to path
            raw_data = {'binary_search': bin_search_mins,
                        'grad_descent': grad_descent_mins}
            raw_data_df = pd.DataFrame(raw_data)

            raw_data_df.to_csv(args["save"])

        elif args['type'] == "html":
            fig.write_html(args['save'])

        elif args['type'] == "img":
            fig.write_image(args['save'])

        elif args['type'] == "json":
            fig.write_json(args["path"])

        else:
            print(
                f"type {args['type']} is not a valid type (html, img, json, raw_data)")

    if args['show']:
        fig.show()
