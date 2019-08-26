import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def stem_plot(data, x_vals=None, title=None, imag=True, ylim=None, xlab=None,
              ylab=None, line=False, f=None):

    if x_vals is None:
        x_vals = np.arange(data.size)

    if ylim is None:
        ylim = (-1, 2.5)

    plt.figure(figsize=(14, 8))
    (markers, stemlines, baseline) = plt.stem(x_vals, np.real(data),
                                              label='Real',
                                              use_line_collection=True)
    plt.setp(stemlines, linestyle="-", color="grey", linewidth=0.5)
    plt.setp(markers, marker='o', color="#19C3F5")
    plt.setp(baseline, linestyle="--", color="grey", linewidth=2)
    if line:
        plt.plot(x_vals, data, linestyle="-", color='#FF4F5B')
    if f is not None:
        xx = np.arange(0, 1, 1/1000.)
        plt.plot(xx, np.sin(2 * np.pi * f * xx), 'g:')

    if imag:

        (markers, stemlines, baseline) = plt.stem(x_vals, np.imag(data),
                                                  label='Imaginary')
        plt.setp(stemlines, linestyle="-", color="grey", linewidth=0.5)
        plt.setp(markers, marker='o', color="#EA8663")
        plt.setp(baseline, linestyle="--", color="grey", linewidth=2)
        plt.legend(loc=1)

    plt.ylim(ylim)
    if not isinstance(xlab, type(None)):
        plt.xlabel(xlab, fontsize=18)
    if not isinstance(ylab, type(None)):
        plt.ylabel(ylab, fontsize=18)
    if not isinstance(title, type(None)):
        plt.title(title, fontsize=20)
    plt.show()


def line_plot(data, title=None, ylim=None, xlab=None):

    plt.figure(figsize=(14, 8))
    plt.plot(data, color='#F76F66')
    plt.plot(np.zeros(data.size), linestyle="--", color="grey")
    if not isinstance(title, type(None)):
        plt.title(title, fontsize=20)
    plt.ylim(ylim)
    if not isinstance(xlab, type(None)):
        plt.xlabel(xlab, fontsize=18)
    plt.show()


def display(data, title='example', shape=None, cmap='gist_stern', vmax=None,
            vmin=None):

    if not isinstance(shape, type(None)):
        data = data.reshape(shape)

    plt.figure(figsize=(14, 8))
    cax = plt.imshow(np.abs(data), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=20)
    plt.colorbar(cax)
    plt.show()


def image(data, title='', cmap='magma', vmax=3500):
    """ Plot Image

    Plot absolute value of input data array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D-array
    title : str, optional
        Image title
    cmap : str, optional
        Colourmap, default is 'magma'
    vmax : int, optional
        Maximum pixel value, default is 3500

    Raises
    ------
    TypeError
        For invalid input data type

    """

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError('Input data must be a 2D numpy array.')

    fig, axis = plt.subplots(1, 1, figsize=(6, 6))
    im = axis.imshow(np.abs(data), cmap=cmap, vmin=0, vmax=vmax)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if title:
        axis.set_title(title, fontsize=20)
    plt.show()


def decomp(data, n_cols=3, cmap='magma'):
    """ Plot Decomposition

    Plot absolute value of decomposed data array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D-array
    n_cols : int, optional
        Number of columns, default is 3
    cmap : str, optional
        Colourmap, default is 'magma'

    Raises
    ------
    TypeError
        For invalid input data type

    """

    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise TypeError('Input data must be a 3D numpy array.')

    n_plots = data.shape[0]
    n_rows = (n_plots - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    vmaxs = [900] * (n_plots - 1) + [3500]
    titles = (['Scale {}'.format(scale) for scale in range(1, n_plots)] +
              ['Coarse Scale'])

    for image, axis, vmax, title in zip(data, axes[:n_plots], vmaxs, titles):
        im = axis.imshow(np.abs(image), cmap=cmap, vmax=vmax)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, ax=axis, cax=cax)
        axis.set_title(title, fontsize=20)

    for axis in axes[n_plots:]:
        axis.axis('off')

    plt.show()


def hist(data, sigma, mean):

    fig, axis = plt.subplots(1, 1, figsize=(9, 6))

    hist = axis.hist(data.ravel(), 25, histtype='step')
    half_max = max(hist[0]) / 2
    axis.axvspan(sigma, sigma + 10, alpha=0.5, color='red', label=r'$\sigma$')
    axis.axvspan(-sigma, -(sigma + 10), alpha=0.5, color='red')
    axis.axvspan(3 * sigma, 3 * sigma + 10, alpha=0.5, color='green',
                 label=r'$3\sigma$')
    axis.axvspan(-3 * sigma, -(3 * sigma + 10), alpha=0.5, color='green')
    axis.axvspan(3 * sigma, 2000, alpha=0.5, color='grey')
    axis.axvspan(-2000, -3 * sigma, alpha=0.5, color='grey')
    axis.legend()
    axis.set_xlim(-2000, 2000)
    axis.set_title(r'$\sigma$ = {:.2f}, $\mu$ = {:.2f}'.format(sigma, mean),
                   fontsize=16)
    plt.show()
