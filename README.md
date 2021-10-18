# Torchopenl3 /Not official fork, changed to work with ebmlibrosa (for embedded platforms)/
TorchopenL3 is an open-source Python library Pytorch Support for computing deep audio embeddings.

Works without librosa (changed to emblibrosa file which is use only few functions which is need for torchopenl3). Excluded numba and resampy packages, which have problem with installation at embedded ARM platforms.
Also it is not used resampy (used only julian for resample audio) and others lubraries which is need numba and others libraries not existed in embedded platforms.

#ATTENTION

works on raspberry very slow - about 17 seconds for 1 sound and mistakes at embeddings calculation - TODO need to fix




[![PyPI](https://img.shields.io/badge/python-3.6%2C%203.7-blue.svg)](https://pypi.python.org/pypi/openl3) [![Build Status](https://travis-ci.org/turian/torchopenl3.png?branch=main)](https://travis-ci.org/turian/torchopenl3) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/turian/torchopenl3/pulse) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/turian/torchopenl3) [![GitHub version](https://badge.fury.io/gh/turian%2Ftorchopenl3.svg)](https://github.com/turian/torchopenl3) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Contributors
![GitHub Contributors Image](https://contrib.rocks/image?repo=turian/torchopenl3)

Please refer to the [Openl3 Library](https://github.com/marl/openl3) for keras version.

The audio and image embedding models provided here are published as part of [1], and are based on the Look, Listen and Learn approach [2]. For details about the embedding models and how they were trained, please see:

[Look, Listen and Learn More: Design Choices for Deep Audio Embeddings](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf)<br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.


# Installation
[![PyPI](https://img.shields.io/badge/python-3.6%2C%203.7-blue.svg)](https://pypi.python.org/pypi/openl3)  
Install via pip 
```
pip install git+https://github.com/toborobot/torchopenl3.git
```

# Using TorchpenL3
```
url = 'https://raw.githubusercontent.com/marl/openl3/master/tests/data/audio/chirp_44k.wav'
filename = 'Sample_audio.wav'
r = requests.get(url, allow_redirects=True)
open(filename, "wb").write(r.content)

emb, ts = torchopenl3.get_audio_embedding(audio, sr)

emb, ts = torchopenl3.get_audio_embedding(audio, sr, content_type="env",
                               input_repr="linear", embedding_size=512)

print(f"Embedding Shape {emb.shape}")
print(f"TimeStamps Shape {ts.shape}")

emb, ts = torchopenl3.get_audio_embedding(audio, sr, center=False)
print(f"Embedding Shape {emb.shape}")
print(f"TimeStamps Shape {ts.shape}")

model = torchopenl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
                                                 embedding_size=512)
emb, ts = torchopenl3.get_audio_embedding(audio, sr, model=model)
print(f"Embedding Shape {emb.shape}")
print(f"TimeStamps Shape {ts.shape}")

```

# Acknowledge
Special Thank you to [Joseph Turian](https://github.com/turain) for his help

[1] [Look, Listen and Learn More: Design Choices for Deep Audio Embeddings](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer\_looklistenlearnmore\_icassp\_2019.pdf)<br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[2] [Look, Listen and Learn](http://openaccess.thecvf.com/content\_ICCV\_2017/papers/Arandjelovic\_Look\_Listen\_and\_ICCV\_2017\_paper.pdf)<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.
