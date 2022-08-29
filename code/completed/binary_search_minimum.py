def func(n):
    # some random quadratic, but feel free to change at will
    return 2*(n**2) - 23*n + 7


arr = [func(x) for x in range(20)]


def binary_search_minimum():
    """
    uses binary search to find the index of an element in an array.
    returns the index of the element in the array or -1 if the element is not present
    """
    # Check base case
    index = len(arr) - 1
    prev_index = 0
    while True:
        if index-prev_index == 1:
            index = prev_index if arr[index] > arr[prev_index] else index
            break

        # the "//" operator will round down if necessary
        half_point = prev_index + ((index - prev_index) // 2)
        first_quartile = prev_index + ((half_point - prev_index) // 2)

        # instead of comparing with x, compare with another number
        if arr[first_quartile] < arr[half_point]:
            index = half_point

        else:
            prev_index = half_point

    return index


index_of_min_point = binary_search_minimum()

print(f"index of minimum element: {index_of_min_point}")
print(f"minimum element in the array: {arr[index_of_min_point]}")

#
# Optional code for plotting the array and the minimum
#
# import plotly.graph_objects as go
#
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=arr, mode="lines+markers", name="array points"))
# fig.add_trace(go.Scatter(x=[index_of_min_point], y=[arr[index_of_min_point]],  # the -1 is because arrays start at 0
#               mode="markers", name="minimum element"))
#
# fig.show()
#
