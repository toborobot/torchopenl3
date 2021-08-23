############################################################################
# custom embedded librosa lib v1.0 20210823
# torchopenl3 testing purpose
############################################################################

import torch
from torch import Tensor
import warnings
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from numpy.lib.stride_tricks import as_strided
import soundfile as sf
import audioread
import samplerate

# Object to hold FFT interfaces
__FFTLIB = None


def set_fftlib(lib=None):
    """Set the FFT library used by librosa.
    Parameters
    ----------
    lib : None or module
        Must implement an interface compatible with `numpy.fft`.
        If ``None``, reverts to `numpy.fft`.
    Examples
    --------
    Use `pyfftw`:
    >>> import pyfftw
    >>> librosa.set_fftlib(pyfftw.interfaces.numpy_fft)
    Reset to default `numpy` implementation
    >>> librosa.set_fftlib()
    """

    global __FFTLIB
    if lib is None:
        from numpy import fft

        lib = fft

    __FFTLIB = lib

# Set the FFT library to numpy's, by default
set_fftlib(None)

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10



def dtype_r2c(d, default=np.complex64):
    """Find the complex numpy dtype corresponding to a real dtype.
    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).
    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.
    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype
    Returns
    -------
    d_c : np.dtype
        The complex dtype
    See Also
    --------
    dtype_c2r
    numpy.dtype
    Examples
    --------
    >>> librosa.util.dtype_r2c(np.float32)
    dtype('complex64')
    >>> librosa.util.dtype_r2c(np.int16)
    dtype('complex64')
    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('complex128')
    """
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(np.float): np.complex,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))

def frame(x, frame_length, hop_length, axis=-1):
    """Slice a data array into (overlapping) frames.
    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.
    However, if the input data is not contiguous in memory, a warning
    will be issued and the output will be a full copy, rather than
    a view of the input data.
    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::
        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]
    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.
    The second way (``axis=0``) results in the array ``x_frames``::
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]
    where each row ``x_frames[i]`` contains a contiguous slice of the input.
    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either to the end of the array (``axis=-1``)
    or the beginning of the array (``axis=0``).
    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : 0 or -1
        The axis along which to frame.
        If ``axis=-1`` (the default), then ``x`` is framed along its last dimension.
        ``x`` must be "F-contiguous" in this case.
        If ``axis=0``, then ``x`` is framed along its first dimension.
        ``x`` must be "C-contiguous" in this case.
    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES) or (N_FRAMES, frame_length, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::
            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]
        If ``axis=0`` (framing on the first dimension), then::
            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]
    Raises
    ------
    ParameterError
        If ``x`` is not an `np.ndarray`.
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.
        If ``hop_length < 1``, frames cannot advance.
        If ``axis`` is not 0 or -1.  Framing is only supported along the first or last axis.
    See Also
    --------
    numpy.asfortranarray : Convert data to F-contiguous representation
    numpy.ascontiguousarray : Convert data to C-contiguous representation
    numpy.ndarray.flags : information about the memory layout of a numpy `ndarray`.
    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)
    >>> frames.shape
    (2048, 1806)
    Or frame along the first axis instead of the last:
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)
    Frame a stereo signal:
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)
    Carve an STFT into fixed-length patches of 32 frames with 50% overlap
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError(
            "Input must be of type numpy.ndarray, " "given type(x)={}".format(type(x))
        )

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)

def pad_center(data, size, axis=-1, **kwargs):
    """Pad an array to a target length along a target axis.
    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`
    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])
    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad`
    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis
    Raises
    ------
    ParameterError
        If ``size < data.shape[axis]``
    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)


def fix_length(data, size, axis=-1, **kwargs):
    """Fix the length an array ``data`` to exactly ``size`` along a target axis.
    If ``data.shape[axis] < n``, pad according to the provided kwargs.
    By default, ``data`` is padded with trailing zeros.
    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, 10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, 5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, 10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
    Parameters
    ----------
    data : np.ndarray
      array to be length-adjusted
    size : int >= 0 [scalar]
      desired length of the array
    axis : int, <= data.ndim
      axis along which to fix length
    kwargs : additional keyword arguments
        Parameters to ``np.pad``
    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        ``data`` either trimmed or padded to length ``size``
        along the specified axis.
    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def get_window(window, Nx, fftbins=True):
    """Compute a window function.
    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:
        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`
    Nx : int > 0
        The length of the window
    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.
    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`
    See Also
    --------
    scipy.signal.get_window
    Notes
    -----
    This function caches at level 10.
    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError(
            "Window size mismatch: " "{:d} != {:d}".format(len(window), Nx)
        )
    else:
        raise ParameterError("Invalid window specification: {}".format(window))

def get_fftlib():
    """Get the FFT library currently used by librosa
    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    """
    global __FFTLIB
    return __FFTLIB

def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
):
    """Short-time Fourier transform (STFT).
    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.
    This function returns a complex-valued matrix D such that
    - ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and
    - ``np.angle(D[f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.
    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.
    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        input signal
    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.
    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.
        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.
        If unspecified, defaults to ``win_length // 4`` (see below).
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.
        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        Defaults to a raised cosine window (`'hann'`), which is adequate for
        most applications in audio signal processing.
        .. see also:: `filters.get_window`
    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.
        .. see also:: `librosa.stream`
    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.
    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="reflect"``),
        ``y`` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If ``center=False``,  this argument is ignored.
        .. see also:: `numpy.pad`
    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.
    See Also
    --------
    istft : Inverse STFT
    reassigned_spectrogram : Time-frequency reassigned spectrogram
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)
    Use left-aligned frames, instead of centered frames
    >>> S_left = librosa.stft(y, center=False)
    Use a shorter hop length
    >>> D_short = librosa.stft(y, hop_length=64)
    Display a spectrogram
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                        ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax)
    >>> ax.set_title('Power spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too large for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels
    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])
    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels
    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies
    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.
    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])
    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz
    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def fft_frequencies(sr=22050, n_fft=2048):
    """Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``
    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    """

    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.
    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.
    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::
        mel = 2595.0 * np.log10(1.0 + f / 700.0).
    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.
    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.
    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.
    .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.
    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc
    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.
    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.
    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.
    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

def mel(
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=np.float32,
):
    """Create a Mel filter-bank.
    This produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_mels    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values
        (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    See also
    --------
    librosa.util.normalize
    Notes
    -----
    This function caches at level 10.
    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])
    Clip the maximum frequency to 8KHz
    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Mel filter', title='Mel filter bank')
    >>> fig.colorbar(img, ax=ax)
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights

#@cache(level=30)
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units
    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.
    Parameters
    ----------
    S : np.ndarray
        input power
    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::
            10 * log10(S / ref)
        Zeros in the output correspond to positions where ``S == ref``.
        If callable, the reference value is computed as ``ref(S)``.
    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``
    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S)) - top_db``
    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude
    Notes
    -----
    This function caches at level 30.
    Examples
    --------
    Get a power spectrogram from a waveform ``y``
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)
    Compute dB relative to peak power
    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)
    Or compare to median power
    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)
    And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead."
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def _spectrogram(
    y=None,
    S=None,
    n_fft=2048,
    hop_length=512,
    power=1,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
):
    """Helper function to retrieve a magnitude spectrogram.
    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.
    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series
    S : None or np.ndarray
        Spectrogram input, optional
    n_fft : int > 0
        STFT window size
    hop_length : int > 0
        STFT hop length
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `filters.get_window`
    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft


# -- Mel spectrogram and MFCCs -- #
def mfcc(
    y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0, **kwargs
):
    """Mel-frequency cepstral coefficients (MFCCs)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(d, t)] or None
        log-power Mel spectrogram

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return

    dct_type : {1, 2, 3}
        Discrete cosine transform (DCT) type.
        By default, DCT type-2 is used.

    norm : None or 'ortho'
        If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an ortho-normal
        DCT basis.

        Normalization is not supported for ``dct_type=1``.

    lifter : number >= 0
        If ``lifter>0``, apply *liftering* (cepstral filtering) to the MFCCs::

            M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter) * lifter / 2)

        Setting ``lifter >= 2 * n_mfcc`` emphasizes the higher-order coefficients.
        As ``lifter`` increases, the coefficient weighting becomes approximately linear.

    kwargs : additional keyword arguments
        Arguments to `melspectrogram`, if operating
        on time series input

    Returns
    -------
    M : np.ndarray [shape=(n_mfcc, t)]
        MFCC sequence

    See Also
    --------
    melspectrogram
    scipy.fftpack.dct

    Examples
    --------
    Generate mfccs from a time series

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[-249.124, -236.652, ..., -619.714, -619.714],
           [  73.787,   51.215, ...,    0.   ,    0.   ],
           ...,
           [ -10.144,   -9.091, ...,    0.   ,    0.   ],
           [ -13.994,  -21.184, ...,    0.   ,    0.   ]],
          dtype=float32)

    Using a different hop length and HTK-style Mel frequencies

    >>> librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True)
    array([[-274.064, -296.403, ..., -643.958, -643.958],
           [  63.888,    0.907, ...,    0.   ,    0.   ],
           ...,
           [  13.069,   36.896, ...,    0.   ,    0.   ],
           [  -2.986,  -13.714, ...,    0.   ,    0.   ]],
          dtype=float32)

    Use a pre-computed log-power Mel spectrogram

    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                    fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.power_to_db(S))
    array([[-222.66 , -209.08 , ..., -627.181, -627.181],
           [  32.214,    2.315, ...,    0.   ,    0.   ],
           ...,
           [   0.872,   -4.195, ...,    0.   ,    0.   ],
           [  29.123,   33.193, ...,    0.   ,    0.   ]],
          dtype=float32)

    Get more components

    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Visualize the MFCC series

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    >>> fig.colorbar(img, ax=ax)
    >>> ax.set(title='MFCC')

    Compare different DCT bases

    >>> m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
    >>> m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
    >>> ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
    >>> fig.colorbar(img, ax=[ax[0]])
    >>> img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
    >>> ax[1].set(title='HTK-style (dct_type=3)')
    >>> fig.colorbar(img2, ax=[ax[1]])
    """

    if S is None:
        S = power_to_db(melspectrogram(y=y, sr=sr, **kwargs))

    M = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc]

    if lifter > 0:
        M *= (
            1
            + (lifter / 2)
            * np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)[
                :, np.newaxis
            ]
        )
        return M
    elif lifter == 0:
        return M
    else:
        raise ParameterError(
            "MFCC lifter={} must be a non-negative number".format(lifter)
        )


def melspectrogram(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
    power=2.0,
    **kwargs,
):
    """Compute a mel-scaled spectrogram.

    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.

    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.

    By default, ``power=2`` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(d, t)]
        spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `librosa.filters.get_window`

    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.

        By default, STFT uses reflection padding.

    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.

    kwargs : additional keyword arguments
        Mel filter bank parameters.

        See `librosa.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram

    See Also
    --------
    librosa.filters.mel
        Mel filter bank construction

    librosa.stft
        Short-time Fourier Transform


    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[3.837e-06, 1.451e-06, ..., 8.352e-14, 1.296e-11],
           [2.213e-05, 7.866e-06, ..., 8.532e-14, 1.329e-11],
           ...,
           [1.115e-05, 5.192e-06, ..., 3.675e-08, 2.470e-08],
           [6.473e-07, 4.402e-07, ..., 1.794e-08, 2.908e-08]],
          dtype=float32)

    Using a pre-computed power spectrogram would give the same result:

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D, sr=sr)

    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S_dB = librosa.power_to_db(S, ref=np.max)
    >>> img = librosa.display.specshow(S_dB, x_axis='time',
    ...                          y_axis='mel', sr=sr,
    ...                          fmax=8000, ax=ax)
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel-frequency spectrogram')
    """

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)



def valid_audio(y, mono=True):
    """Determine whether a variable contains valid audio data.
    If ``mono=True``, then ``y`` is only considered valid if it has shape
    ``(N,)`` (number of samples).
    If ``mono=False``, then ``y`` may be either monophonic, or have shape
    ``(2, N)`` (stereo) or ``(K, N)`` for ``K>=2`` for general multi-channel.
    Parameters
    ----------
    y : np.ndarray
      The input data to validate
    mono : bool
      Whether or not to require monophonic audio
    Returns
    -------
    valid : bool
        True if all tests pass
    Raises
    ------
    ParameterError
        In any of these cases:
            - ``type(y)`` is not ``np.ndarray``
            - ``y.dtype`` is not floating-point
            - ``mono == True`` and ``y.ndim`` is not 1
            - ``mono == False`` and ``y.ndim`` is not 1 or 2
            - ``mono == False`` and ``y.ndim == 2`` but ``y.shape[0] == 1``
            - ``np.isfinite(y).all()`` is False
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> # By default, valid_audio allows only mono signals
    >>> filepath = librosa.ex('trumpet', hq=True)
    >>> y_mono, sr = librosa.load(filepath, mono=True)
    >>> y_stereo, _ = librosa.load(filepath, mono=False)
    >>> librosa.util.valid_audio(y_mono), librosa.util.valid_audio(y_stereo)
    True, False
    >>> # To allow stereo signals, set mono=False
    >>> librosa.util.valid_audio(y_stereo, mono=False)
    True
    See also
    --------
    numpy.float32
    """

    if not isinstance(y, np.ndarray):
        raise ParameterError("Audio data must be of type numpy.ndarray")

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError("Audio data must be floating-point")

    if mono and y.ndim != 1:
        raise ParameterError(
            "Invalid shape for monophonic audio: "
            "ndim={:d}, shape={}".format(y.ndim, y.shape)
        )

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError(
            "Audio data must have shape (samples,) or (channels, samples). "
            "Received shape={}".format(y.shape)
        )

    elif y.ndim == 2 and y.shape[0] < 2:
        raise ParameterError(
            "Mono data must have shape (samples,). " "Received shape={}".format(y.shape)
        )

    if not np.isfinite(y).all():
        raise ParameterError("Audio buffer is not finite everywhere")

    return True

def to_mono(y):
    """Convert an audio signal to mono by averaging samples across channels.
    Parameters
    ----------
    y : np.ndarray [shape=(2,n) or shape=(n,)]
        audio time series, either stereo or mono
    Returns
    -------
    y_mono : np.ndarray [shape=(n,)]
        ``y`` as a monophonic time-series
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> y_mono = librosa.to_mono(y)
    >>> y_mono.shape
    (117601,)
    """
    # Ensure Fortran contiguity.
    y = np.asfortranarray(y)

    # Validate the buffer.  Stereo is ok here.
    valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


#@cache(level=20)
def resample(
    y, orig_sr, target_sr, res_type="linear", fix=True, scale=False, **kwargs
):
    """Resample a time series from orig_sr to target_sr
    By default, this uses a linear method
    for band-limited sinc interpolation.  The alternate ``res_type`` values listed below
    offer different trade-offs of speed and quality.
    20210809 made changes to delete resampy wich need numba for execution
    numba cant be installed to raspberry 64 bit os Now
    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.
    orig_sr : number > 0 [scalar]
        original sampling rate of ``y``
    target_sr : number > 0 [scalar]
        target sampling rate
    res_type : str
        resample type
        'fft' or 'scipy'
            `scipy.signal.resample` Fourier method.
        'polyphase'
            `scipy.signal.resample_poly` polyphase filtering. (fast)
        'linear'
            `samplerate` linear interpolation. (very fast)
        'zero_order_hold'
            `samplerate` repeat the last value between samples. (very fast)
        'sinc_best', 'sinc_medium' or 'sinc_fastest'
            `samplerate` high-, medium-, and low-quality sinc interpolation.
        'soxr_vhq', 'soxr_hq', 'soxr_mq' or 'soxr_lq'
            `soxr` Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation.
            ``'soxr_hq'`` is the default setting of `soxr` (fast)
        'soxr_qq'
            `soxr` Quick cubic interpolation (very fast)
        .. note::
            `samplerate` and `soxr` are not installed with `librosa`.
            To use `samplerate` or `soxr`, they should be installed manually::
                $ pip install samplerate
                $ pip install soxr
        .. note::
            When using ``res_type='polyphase'``, only integer sampling rates are
            supported.
    fix : bool
        adjust the length of the resampled signal to be of size exactly
        ``ceil(target_sr * len(y) / orig_sr)``
    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately
        equal total energy.
    kwargs : additional keyword arguments
        If ``fix==True``, additional keyword arguments to pass to
        `librosa.util.fix_length`.
    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        ``y`` resampled from ``orig_sr`` to ``target_sr``
    Raises
    ------
    ParameterError
        If ``res_type='polyphase'`` and ``orig_sr`` or ``target_sr`` are not both
        integer-valued.
    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy
    samplerate.converters.resample
    soxr.resample
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    Downsample from 22 KHz to 8 KHz
    >>> y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((117601,), (42668,))
    """

    # First, validate the audio buffer
    valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError(
                "polyphase resampling is only supported for integer-valued sampling rates."
            )

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        import samplerate

        # We have to transpose here to match libsamplerate
        y_hat = samplerate.resample(y.T, ratio, converter_type=res_type).T
    elif res_type.startswith('soxr'):
        import soxr

        # We have to transpose here to match soxr
        y_hat = soxr.resample(y.T, orig_sr, target_sr, quality=res_type).T
    #else:
    #    y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)


def __audioread_load(path, offset, duration, dtype):
    """Load an audio buffer using audioread.
    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native

def load(
    path,
    sr=22050,
    mono=True,
    offset=0.0,
    duration=None,
    dtype=np.float32,
    res_type="kaiser_best",
):
    """Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate
    (default ``sr=22050``).
    To preserve the native sampling rate of the file, use ``sr=None``.
    Parameters
    ----------
    path : string, int, pathlib.Path or file-like object
        path to the input file.
        Any codec supported by `soundfile` or `audioread` will work.
        Any string file paths, or any object implementing Python's
        file interface (e.g. `pathlib.Path`) are supported as `path`.
        If the codec is supported by `soundfile`, then `path` can also be
        an open file descriptor (int).
        On the contrary, if the codec is not supported by `soundfile`
        (for example, MP3), then `path` must be a file path (string or `pathlib.Path`).
    sr   : number > 0 [scalar]
        target sampling rate
        'None' uses the native sampling rate
    mono : bool
        convert signal to mono
    offset : float
        start reading after this time (in seconds)
    duration : float
        only load up to this much audio (in seconds)
    dtype : numeric type
        data type of ``y``
    res_type : str
        resample type (see note)
        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').
            For alternative resampling modes, see `resample`
        .. note::
           `audioread` may truncate the precision of the audio data to 16 bits.
           See :ref:`ioformats` for alternate loading methods.
    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series
    sr   : number > 0 [scalar]
        sampling rate of ``y``
    Examples
    --------
    >>> # Load an ogg vorbis file
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename)
    >>> y
    array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05],
          dtype=float32)
    >>> sr
    22050
    >>> # Load a file and resample to 11 KHz
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([-8.746e-04, -3.363e-04, ..., -1.301e-05,  0.000e+00],
          dtype=float32)
    >>> sr
    11025
    >>> # Load 5 seconds of a file, starting 15 seconds in
    >>> filename = librosa.ex('brahms')
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([0.146, 0.144, ..., 0.128, 0.015], dtype=float32)
    >>> sr
    22050
    """

    try:
        with sf.SoundFile(path) as sf_desc:
            sr_native = sf_desc.samplerate
            if offset:
                # Seek to the start of the target read
                sf_desc.seek(int(offset * sr_native))
            if duration is not None:
                frame_duration = int(duration * sr_native)
            else:
                frame_duration = -1

            # Load the target number of frames, and transpose to match librosa form
            y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    except RuntimeError as exc:
        # If soundfile failed, try audioread instead
        if isinstance(path, (str, pathlib.PurePath)):
            warnings.warn("PySoundFile failed. Trying audioread instead.")
            y, sr_native = __audioread_load(path, offset, duration, dtype)
        else:
            raise (exc)

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)

    if sr is not None:
        y = resample(y, sr_native, sr, res_type=res_type)

    else:
        sr = sr_native

    return y, sr

############################################################################
# end custom embedded librosa lib
############################################################################
