import numpy as np

def rolling_window(input_array, size_kernel, stride, print_dims = True):
    """Function to get rolling windows.

    Arguments:
        input_array {numpy.array} -- Input, by default it only works with depth equals to 1. 
                                      It will be treated as a (height, width) image. If the input have (height, width, channel) 
                                      dimensions, it will be rescaled to two-dimension (height, width)
        size_kernel {int} -- size of kernel to be applied. Usually 3,5,7. It means that a kernel of (size_kernel, size_kernel) will be applied
                             to the image.
        stride {int or tuple} -- horizontal and vertical displacement

    Keyword Arguments:
        print_dims {bool} -- [description] (default: {True})

    Returns:
        [list] -- A list with the resulting numpy.arrays 
    """   
    # Check right input dimension
    assert(len(input_array.shape) in set([1,2])), "input_array must have dimension 2 or 3. Yours have dimension {}".format(len(input_array))

    if input_array.shape == 3:
        input_array = input_array[:,:,0]

    # Stride: horizontal and vertical displacement
    if isinstance(stride,int):
        sh, sw = stride, stride
    elif isinstance(stride,tuple):
        sh, sw = stride

    # Input dimension (height, width)
    n_ah, n_aw = input_array.shape

    # Filter dimension (or window)
    n_k  = size_kernel

    dim_out_h = int(np.floor( (n_ah - n_k) / sh + 1 ))
    dim_out_w = int(np.floor( (n_aw - n_k) / sw + 1 ))

    # List to save output arrays
    list_tensor = []

    # Initialize row position
    start_row = 0
    for i in range(dim_out_h):
        start_col = 0
        for j in range(dim_out_w):
            sub_array = []
            # Get one window
            sub_array.append(input_array[start_row:(start_row+n_k), start_col:(start_col+n_k)])
            sub_array.append(start_row)
            sub_array.append(start_col)
            # Append sub_array
            list_tensor.append(sub_array)
            start_col += sw
        start_row += sh

    if print_dims: 
        print("- Input tensor dimensions -- ", input_array.shape)
        print("- Kernel dimensions -- ", (n_k, n_k))
        print("- Stride (h,w) -- ", (sh, sw))
        print("- Total windows -- ", len(list_tensor))

    return list_tensor