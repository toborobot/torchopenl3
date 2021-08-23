# Torchopenl3 /Not official fork, changed to work with ebmlibrosa (for embedded platforms)/
TorchopenL3 is an open-source Python library Pytorch Support for computing deep audio embeddings.


[![PyPI](https://img.shields.io/badge/python-3.6%2C%203.7-blue.svg)](https://pypi.python.org/pypi/openl3) [![Build Status](https://travis-ci.org/turian/torchopenl3.png?branch=main)](https://travis-ci.org/turian/torchopenl3) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/turian/torchopenl3/pulse) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/turian/torchopenl3) [![GitHub version](https://badge.fury.io/gh/turian%2Ftorchopenl3.svg)](https://github.com/turian/torchopenl3) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Contributors
![GitHub Contributors Image](https://contrib.rocks/image?repo=turian/torchopenl3)

Please refer to the [Openl3 Library](https://github.com/marl/openl3) for keras version.

The audio and image embedding models provided here are published as part of [1], and are based on the Look, Listen and Learn approach [2]. For details about the embedding models and how they were trained, please see:

[Look, Listen and Learn More: Design Choices for Deep Audio Embeddings](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf)<br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

# Comparasion

We run torchopenl3 over 100 audio files and compare with openl3 embeddings. Below is the MAE (Mean Absolute Error) Table


| Content_type | Input_repr | Emd_size |           MAE          |
|:------------:|:----------:|:--------:|:----------------------:|
|      Env     |   Linear   |    512   | 1.1522600237867664e-06 |
|      Env     |   Linear   |   6144   |  1.027089645617707e-06 |
|      Env     |   Mel128   |    512   | 1.2094695046016568e-06 |
|      Env     |   Mel128   |   6144   | 1.0968088741947213e-06 |
|      Env     |   Mel256   |    512   | 1.1641358707947802e-06 |
|      Env     |   Mel256   |   6144   | 1.0069775197507625e-06 |
|     Music    |   Linear   |    512   |  1.173499645119591e-06 |
|     Music    |   Linear   |   6144   |  1.048712784381678e-06 |
|     Music    |   Mel128   |    512   | 1.1837427564387327e-06 |
|     Music    |   Mel128   |   6144   | 1.0497348176841115e-06 |
|     Music    |   Mel256   |    512   | 1.1619711483490392e-06 |
|     Music    |   Mel256   |   6144   |  9.881532906774738e-07 |


# Installation
[![PyPI](https://img.shields.io/badge/python-3.6%2C%203.7-blue.svg)](https://pypi.python.org/pypi/openl3)  
Install via pip 
```
pip install git+https://github.com/turian/torchopenl3.git
```
Install the package with all dev libraries (i.e. tensorflow openl3)
```
git clone https://github.com/turian/torchopenl3.git
pip3 install -e ".[dev]"
```

Install Docker and work within the Docker environment.
Unfortunately this Docker image is quite big (about 4 GB) because

```
docker pull turian/torchopenl3
# Or, build the docker yourself
#docker build -t turian/torchopenl3 .
```

# Using TorchpenL3
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LHbM1WN_1LK61R6XEbUlrtmRb3fKlDov?usp=sharing) 

To help you get started with TorchopenL3 please go through the colab file.

# Acknowledge
Special Thank you to [Joseph Turian](https://github.com/turain) for his help

[1] [Look, Listen and Learn More: Design Choices for Deep Audio Embeddings](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer\_looklistenlearnmore\_icassp\_2019.pdf)<br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[2] [Look, Listen and Learn](http://openaccess.thecvf.com/content\_ICCV\_2017/papers/Arandjelovic\_Look\_Listen\_and\_ICCV\_2017\_paper.pdf)<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.
