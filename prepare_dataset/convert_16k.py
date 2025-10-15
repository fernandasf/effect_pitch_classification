import numpy as np
import IPython.display as ipd
import glob
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from tqdm.contrib.concurrent import process_map

def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    reference: https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def convert_16k(file):
    audio = librosa.load(file, sr=48000)[0]
    audio_16k = librosa.resample(audio, orig_sr=48000, target_sr=16000)
    audio_int16 = float2pcm(audio_16k, dtype='int16')
    out_name = file.replace("fernanda_audios", "GPS_cmd_16k")
    wavfile.write(out_name, 16000, audio_int16)

if '__main__' == __name__:
    output = "/home/jovyan/work/OneDrive/Documentos/Doutorado/dataset/GPS_cmd_16k"
    os.makedirs(output, exist_ok=True)
    data = glob.glob("/home/jovyan/work/OneDrive/Documentos/Doutorado/dataset/fernanda_audios/*.wav")
    print("len(data): ", len(data))
    
    results = process_map(convert_16k, data, max_workers=8, chunksize=1, desc="Processing")


