# ma-ot-monotone

This repository contains the Python code that can be used to reproduce the numerical experiments described in the paper:

G. Bonnet and J.‑M. Mirebeau. [Monotone discretization of the Monge-Ampère equation of optimal transport](https://www.esaim-m2an.org/articles/m2an/abs/2022/03/m2an210105/m2an210105.html). *ESAIM: Mathematical Modelling and Numerical Analysis*, 56(3):815–865, 2022.

This code is made available mainly for reproducibility purposes, and we do not intend to maintain it in the future.

Be aware that the code was not designed to be particularly readable or self-explanatory. Nevertheless, we offer a limited description of the interface of the code in the file [main/demo.py](main/demo.py), in which we demonstrate how to use it in order to solve a simple optimal transport problem. The implementation of the numerical method itself can be found in the file [main/ma.py](main/ma.py).

## Dependencies

The Python packages that the code depends on are the following:

- [numpy](https://pypi.org/project/numpy/), [scipy](https://pypi.org/project/scipy/), and [matplotlib](https://pypi.org/project/matplotlib/).
- [agd](https://pypi.org/project/agd/), which corresponds to the [AdaptiveGridDiscretizations](https://github.com/Mirebeau/AdaptiveGridDiscretizations) GitHub project by the second author of the paper.
- [imageio](https://pypi.org/project/imageio/) and [lz4](https://pypi.org/project/lz4/), solely for numerical experiments related to the far field refractor problem in nonimaging optics.

Numerical experiments related to nonimaging optics also depend on the [appleseed](https://appleseedhq.net/) rendering engine.

### Installing the dependencies, and reproducibility of the Python environment

The above Python packages can be installed using either the `pip` command or the [Poetry](https://python-poetry.org/) dependency management system:

```console
$ pip install numpy scipy matplotlib agd imageio lz4
```

or

```console
$ poetry install
```

While the first option is simpler, the second one allows to use the exact versions of those Python packages that the code was tested with.

The version of Python itself that was used when testing the code was Python 3.10.7.

Binaries for the appleseed rendering engine can be found on [the appleseed website](https://appleseedhq.net/). The code was tested with appleseed version 2.1.0-beta. Remember that appleseed only needs to be installed if you intend to run the numerical experiments related to nonimaging optics.

## Reproducing the numerical experiments

In order to reproduce only the numerical experiments that are not related to nonimaging optics, run:

```console
$ make all-but-optics
```

If you want to also reproduce the numerical experiments that are related to nonimaging optics, you may need to specify the path to the `appleseed.cli` binary on the command line:

```console
$ make optics APPLESEED_CLI=path/to/appleseed.cli
```

or

```console
$ make all APPLESEED_CLI=path/to/appleseed.cli
```

The results of the numerical experiments will be saved in the directory [figures/out](figures/out). In order to know how to run a specific numerical experiment, you can follow the rules from [the Makefile](Makefile).
