arr = [x + 1 for x in range(20)]  # 1, 2, 3, 4, 5 ... 20


def binary_search(x):
    """
    uses binary search to find the index of an element in an array.
    returns the index of the element in the array or -1 if the element is not present
    """
    # Check base case
    index = len(arr) - 1
    prev_index = 0
    if arr[index] > x:
        while True:
            # the "//" operator will round down if necessary
            half_point = prev_index + ((index - prev_index) // 2)
            if arr[half_point] == x:
                return half_point

            elif x < arr[half_point]:
                index = half_point

            else:
                prev_index = half_point
    else:
        # Element is not present in the array
        return -1


print(binary_search(16))
