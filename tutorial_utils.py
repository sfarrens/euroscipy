import numpy as np
import matplotlib
from sf_tools.image.stamp import FetchStamps
matplotlib.use("Agg")
try:
    from pysap import load_transform
except ImportError:
    from astropy.convolution import convolve_fft
    from modopt.interface.errors import warn
    warn('PySAP not installed, using backup filters.')
    import_pysap = False
else:
    import_pysap = True


def convolve(data, kernel):
    """ Convolve

    Convolve input data with kernel.

    Parameters
    ----------
    data : np.ndarray
        Input 2D data array
    kernel : np.ndarray
        Input 2D kernel array

    Returns
    -------
    np.ndarray
        Convolved array

    Raises
    ------
    TypeError
        For invalid input data type
    TypeError
        For invalid input kernel type

    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    if not isinstance(kernel, np.ndarray) or data.ndim != 2:
        raise TypeError('Input kernel must be a 2D numpy array.')

    return convolve_fft(data, kernel, boundary='wrap', crop=False,
                        nan_treatment='fill', normalize_kernel=False)


def decompose(data, n_scales=4):
    """ Decompose

    Obtain the wavelet decomposition of the input date using an isotropic
    undecimated wavelet transform.

    Parameters
    ----------
    data : np.ndarray
        Input 2D-array
    n_scales : int, optional
        Number of wavelet scales, default is 4

    Returns
    -------
    np.ndarray
        Wavelet decomposition 3D-array

    Raises
    ------
    TypeError
        For invalid input data type
    TypeError
        For invalid input n_scales type

    Examples
    --------
    >>> import numpy as np
    >>> from tutorial_utils import decompose
    >>> np.random.seed(0)
    >>> data = np.random.ranf((3, 3))
    >>> decompose(data)
    array([[[-0.06020004,  0.09427285, -0.03005594],
            [-0.06932276, -0.21794325, -0.02309608],
            [-0.22873539,  0.17666274,  0.19976479]],

           [[-0.04426706, -0.02943552, -0.01460403],
            [-0.0475564 , -0.01650959,  0.01453722],
            [-0.0240097 ,  0.02943558,  0.08288085]],

           [[-0.0094105 , -0.0110383 , -0.01266617],
            [-0.00393927, -0.00619102, -0.00844282],
            [ 0.01415205,  0.0110383 ,  0.00792474]],

           [[ 0.66269112,  0.6613903 ,  0.66008949],
            [ 0.66570163,  0.66429865,  0.6628958 ],
            [ 0.67618024,  0.67463636,  0.67309237]]])

    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    if not isinstance(n_scales, int) or n_scales < 1:
        raise TypeError('n_scales must be a positive integer.')

    if import_pysap:

        trans_name = 'BsplineWaveletTransformATrousAlgorithm'
        trans = load_transform(trans_name)(nb_scale=n_scales,
                                           padding_mode="symmetric")
        trans.data = data
        trans.analysis()

        res = np.array(trans.analysis_data, dtype=np.float)

    else:

        filters = np.load('data/filters_{}.npy'.format(n_scales))
        res = np.array([convolve(data, f) for f in filters])

    return res


def recombine(data):
    """ Recombine

    Recombine wavelet decomposition.

    Parameters
    ----------
    data : np.ndarray
        Input 3D-array

    Returns
    -------
    np.ndarray
        Recombined 2D-array

    Raises
    ------
    TypeError
        For invalid input data type

    Examples
    --------
    >>> import numpy as np
    >>> from tutorial_utils import recombine
    >>> np.random.seed(0)
    >>> data = np.random.ranf((4, 3, 3))
    >>> recombine(data)
    array([[2.65508069, 2.89877487, 2.52493858],
           [2.17664192, 2.58496449, 1.95360968],
           [1.21142489, 1.57070222, 2.55727139]])

    """

    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise TypeError('Input data must be a 3D numpy array.')

    return np.sum(data, axis=0)


def sigma_clip(data, n_iter=3):
    """ Sigma Clipping

    Perform iterative sigma clipping on input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    n_iter : int, optional
        Number of iterations, default is 3

    Returns
    -------
    tuple
        mean and standard deviation of clipped sample

    Raises
    ------
    TypeError
        For invalid input data type
    TypeError
        For invalid input n_iter type

    Examples
    --------
    >>> import numpy as np
    >>> from tutorial_utils import sigma_clip
    >>> np.random.seed(0)
    >>> data = np.random.ranf((3, 3))
    >>> sigma_clip(data)
    (0.6415801460355164, 0.17648980804276407)

    """

    if not isinstance(data, np.ndarray):
        raise TypeError('Input data must be a numpy array.')

    if not isinstance(n_iter, int) or n_iter < 1:
        raise TypeError('n_iter must be a positive integer.')

    for _iter in range(n_iter):
        if _iter == 0:
            clipped_data = data
        else:
            clipped_data = data[np.abs(data - mean) < (3 * sigma)]
        mean = np.mean(clipped_data)
        sigma = np.std(clipped_data)

    return mean, sigma


def noise_est(data, n_iter=3):
    """ Noise Estimate

    Estimate noise standard deviation of input data using smoothed median.

    Parameters
    ----------
    data : np.ndarray
        Input 2D-array
    n_iter : int, optional
        Number of sigma clipping iterations, default is 3

    Returns
    -------
    float
        Noise standard deviation

    Raises
    ------
    TypeError
        For invalid input data type

    Examples
    --------
    >>> import numpy as np
    >>> from tutorial_utils import noise_est
    >>> np.random.seed(0)
    >>> data = np.random.ranf((3, 3))
    >>> noise_est(data)
    0.11018895815851695

    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    ft_obj = FetchStamps(data, pixel_rad=1, all=True, pad_mode='edge')
    median = ft_obj.scan(np.median).reshape(data.shape)
    mean, sigma = sigma_clip(data - median, n_iter)

    correction_factor = 0.972463

    return sigma / correction_factor


def sigma_scales(sigma, n_scales=4, kernel_shape=(51, 51)):
    """ Sigma Scales

    Get rescaled sigma values for wavelet decomposition.

    Parameters
    ----------
    sigma : float
        Noise standard deviation
    n_scales : int, optional
        Number of wavelet scales, default is 4
    kernel_shape : tuple, list or np.ndarray, optional
        Shape of dummy image kernel

    Returns
    -------
    np.ndarray
        Rescaled sigma values not including coarse scale

    Raises
    ------
    TypeError
        For invalid sigma type
    TypeError
        For invalid kernel_shape type

    Examples
    --------
    >>> from tutorial_utils import sigma_scales
    >>> sigma_scales(1)
    array([0.89079631, 0.20066385, 0.0855075 ])

    """

    if not isinstance(sigma, (int, float)):
        raise TypeError('Input sigma must be an int or a float.')

    if not isinstance(kernel_shape, (tuple, list, np.ndarray)):
        raise TypeError('kernel_shape must be a tuple, list or numpy array.')

    kernel_shape = np.array(kernel_shape)
    kernel_shape += kernel_shape % 2 - 1

    dirac = np.zeros(kernel_shape, dtype=float)
    dirac[tuple(zip(kernel_shape // 2))] = 1.

    return float(sigma) * np.linalg.norm(decompose(dirac, n_scales=n_scales),
                                         axis=(1, 2))[:-1]
