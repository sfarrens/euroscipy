<a href="http://www.cosmostat.org/" target="_blank"><img align="left" width="300" src="http://www.cosmostat.org/wp-content/uploads/2017/07/CosmoStat-Logo_WhiteBK-e1499155861666.png" alt="CosmoStat Logo"></a>
<br>
<br>
<br>
<br>
<br>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sfarrens/euroscipy/master)
<br>
# Astrophysical Image Processing

<br>

<img align="left" src="https://www.euroscipy.org/theme/images/euroscipy_logo.png" width="50">

This tutorial was prepared for the [EuroScipy 2019](https://www.euroscipy.org/2019/) conference. The objective being to provide a brief introduction to some of the methods used to analyse astronomical images, in particular the use of sparsity to denoise images.
<br>
<br>

Please see the [CosmoStat website](http://www.cosmostat.org/tutorials) for more signal and image processing tutorials.

## Contents
---

1. [Requirements](#Requirements)
1. [Notebooks](#Notebooks)

## Requirements
---

It is recommended to run this tutorial via the interactive online [Binder](https://mybinder.org/v2/gh/sfarrens/euroscipy/master) interface. To run it locally, however, the following requirements need to be installed.

1. [Python](https://www.python.org/) >=3.5
1. [astropy](https://www.astropy.org/) >=3.2.1
1. [jupyter](https://jupyter.org/) >=1.0.0
1. [matplotlib](https://matplotlib.org/) >=3.1.1
1. [modopt](https://cea-cosmic.github.io/ModOpt/) >=1.4.0
1. [numpy](https://www.numpy.org/) >=1.16.4
1. [PySAP](https://github.com/CEA-COSMIC/pysap)
1. [sf_tools](https://github.com/sfarrens/sf_tools) >=2.0.4

All of the requirements can be built with the provided conda enviroment.

```bash
conda env create -f enviroment.yml
```

> Note that PySAP requires cmake and a C++ compiler that supports OpenMP. See [here](https://github.com/CEA-COSMIC/pysap/blob/master/doc/macos_install.rst) for macOS help.

## Notebooks
---

1. [Introduction to Sparsity](./sparsity.ipynb)
1. [Astronomical Image Denoising](./denoising.ipynb)
