<div align="center">
  <img width="300px" src="https://pytorch.org/static/img/logos/pytorch-logo-dark.png" alt="gans with pytorch"/>
</div>

Generative adversarial networks using Pytorch
=======================================

## Table of Contents
  - [pix2pix](#pix2pix)
  - [relativistic gan](#ralsgan)
  - [super resolution gan](#srgan)
    - [Ignore Whitespace](#ignore-whitespace)
    - [Adjust Tab Space](#adjust-tab-space)
    - [Commit History by Author](#commit-history-by-author)
    - [Cloning a Repository](#cloning-a-repository)
    - [Branch](#branch)
      - [Compare all Branches to Another Branch](#compare-all-branches-to-another-branch)
      - [Comparing Branches](#comparing-branches)

[![Build Status](https://api.travis-ci.org/python/mypy.svg?branch=master)](https://travis-ci.org/python/mypy)
[![Chat at https://gitter.im/python/typing](https://badges.gitter.im/python/typing.svg)](https://gitter.im/python/typing?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

Ok but why? 
----------------------------------

Any help in testing, development, documentation and other tasks is
highly appreciated and useful to the project. There are tasks for
contributors of all experience levels. If you're just getting started,
ask on the [gitter chat](https://gitter.im/python/typing) for ideas of good
beginner issues.

Setting it up 
----------------------------------

I started doing this work with <b>[Pytorch](https://pytorch.org/) 0.4.0</b> and <b>Python 3.6</b> 
(with <b>[Cuda](https://developer.nvidia.com/cuda-downloads) 9.0</b> and 
<b>[CuDNN](https://developer.nvidia.com/cudnn) 7</b>), with <b>Ubuntu 16.04</b>. 
Around right after "SRGAN"s, I switched to <b>Pytorch 0.4.1</b>, 
<b>Cuda 9.2</b> and <b>[CuDNN](https://developer.nvidia.com/cudnn) 7.2</b>. 
For visualizing the GAN generation progress on your browser, you will need the facebook's <b>[visdom](https://github.com/facebookresearch/visdom)</b> library.
I recommend using [anaconda3](https://conda.io/docs/user-guide/install/download.html) to install dependencies and 
<b>[Pycharm](https://www.jetbrains.com/pycharm/)</b> community version to edit the code.
For dataset, I provide either scripts or links. Although I prefer using datasets included in
Pytorch whenever I can for simplicity, there is only so much you can do with digits or CIFAR images. Still, I stick to 
previously used datasets to cut off my implementation time, where the data acquisition and preparation takes easily 
more than 60-70% of the time.

SRGAN
-------------
### pix2pix
> **_[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)_** 

[[Code](https://github.com/ozanciga/gans-with-pytorch/tree/master/pix2pix)]

**_Quick summary_**: Use a reference image (can be a annotation mask, a natural scene image that we want to turn the daytime into night, or even sketches of shoes), and change it to a target image. Here, we see the labels of roads, buildings, people etc turned into actual cityscapes just from crude and undetailed simple color masks. 

<center>
<img src="https://raw.githubusercontent.com/ozanciga/gans-with-pytorch/master/pix2pix/out/GAN%20output%20%5BEpoch%20216%5D.jpg" alt="Network architecture" />
<img src="https://raw.githubusercontent.com/ozanciga/gans-with-pytorch/master/pix2pix/out/GAN%20output%20%5BEpoch%20240%5D.jpg" alt="Network architecture" />
</center>

<div align="right"> <b><a href="#">∧ Go to top</a></b> </div>

### ralsgan
**[The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734)**
<center>
<img src="https://raw.githubusercontent.com/ozanciga/gans-with-pytorch/master/ralsgan/out/GAN%20output%20%5BIteration%2031680%5D.png" alt="Network architecture" />
<img src="https://raw.githubusercontent.com/ozanciga/gans-with-pytorch/master/ralsgan/out/GAN%20output%20%5BIteration%2031616%5D.png" alt="Network architecture" />
</center>

Quick summary: Unlike any previous model, this GAN is able to generate high resolution images (up to 256 x 256) from scratch relatively fast. 
Previously, people either stuck to resolutions as low as 64 x 64, or they have [progressively increased the resolution](https://arxiv.org/abs/1710.10196) which
takes a long time.



### srgan
> **[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)**
<center><img src="https://cdn-images-1.medium.com/max/1800/1*zsiBj3IL4ALeLgsCeQ3lyA.png" alt="Network architecture" /></center>

Quick summary: Increase the resolution of images.

You can mix dynamic and static typing in your programs. You can always
fall back to dynamic typing when static typing is not convenient, such
as for legacy code.

Here is a small example to whet your appetite (Python 3):

```python
from typing import Iterator

def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b
```
See [the documentation](http://mypy.readthedocs.io/en/stable/introduction.html) for more examples.

For Python 2.7, the standard annotations are written as comments:
```python
def is_palindrome(s):
    # type: (str) -> bool
    return s == s[::-1]
```

See [the documentation for Python 2 support](http://mypy.readthedocs.io/en/latest/python2.html).

Mypy is in development; some features are missing and there are bugs.
See 'Development status' below.

Requirements
------------

You need Python 3.4 or later to run mypy.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux flavors, OS X and Windows, packages are available at

  http://www.python.org/getit/


Quick start
-----------

Mypy can be installed using pip:

    $ python3 -m pip install -U mypy

If you want to run the latest version of the code, you can install from git:

    $ python3 -m pip install -U git+git://github.com/python/mypy.git


Now, if Python on your system is configured properly (else see
"Troubleshooting" below), you can type-check the [statically typed parts] of a
program like this:

    $ mypy PROGRAM

You can always use a Python interpreter to run your statically typed
programs, even if they have type errors:

    $ python3 PROGRAM

[statically typed parts]: http://mypy.readthedocs.io/en/latest/basics.html#function-signatures

Troubleshooting
---------------

Depending on your configuration, you may have to run `pip` like
this:

    $ python3 -m pip install -U mypy

This should automatically install the appropriate version of
mypy's parser, typed-ast.  If for some reason it does not, you
can install it manually:

    $ python3 -m pip install -U typed-ast

If the `mypy` command isn't found after installation: After
`python3 -m pip install`, the `mypy` script and
dependencies, including the `typing` module, will be installed to
system-dependent locations.  Sometimes the script directory will not
be in `PATH`, and you have to add the target directory to `PATH`
manually or create a symbolic link to the script.  In particular, on
Mac OS X, the script may be installed under `/Library/Frameworks`:

    /Library/Frameworks/Python.framework/Versions/<version>/bin

In Windows, the script is generally installed in
`\PythonNN\Scripts`. So, type check a program like this (replace
`\Python34` with your Python installation path):

    C:\>\Python34\python \Python34\Scripts\mypy PROGRAM

