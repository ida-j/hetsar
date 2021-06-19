# HETSAR: Python package to estimate spatial autoregressive models with heterogeneous coefficients

hetsar fits spatial autoregressive panel data models with heterogeneous coefficients. 
The estimation is performed via quasi maximum-likelihood.
See Aquaro, Bailey and Pesaran (J. Appl. Econometrics, 2021) for technical details.

The hetsar package was written in 2021 by Ida Johnsson. It is distributed under the 3-Clause BSD license.

See the source for this project here:
<https://github.com/ida-j/hetsar>.


# Installation

## Prerequisites

Install dependencies:
```
$ pip install -r requirements.txt
```

Install from pip:

```
$ pip install hetsar
```

Install from source:

```
$ git clone https://github.com/ida-j/hetsar.git
$ cd hetsar
$ python setup.py build_ext --inplace
$ python setup.py install
