def matrix_formatter(mtrx, dim: int, precision: int = 4):
    '''
    Format the matrix output. Otherwise, some of the elements are float, some are np.float64
    :param mtrx: The matrix to be formatted.
    :param dim: The dimension (rows) of the matrix.
    :param precision: The number of digits of the matrix elements that you want to see.
    :return: A nice matrix output
    '''
    if dim < 1:
        return mtrx
    elif dim == 1:
        return [round(float(elm), precision) for elm in mtrx]
    else:
        return [matrix_formatter(elm, dim - 1, precision) for elm in mtrx]