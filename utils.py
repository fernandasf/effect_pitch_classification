import numpy as np
#import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd

class UtilsIO():
    def __init__(self):
        pass

    @staticmethod
    def load_audio(path, verbose=False):
        y = librosa.load(path, sr=16000)[0]
        if verbose:
            ipd.display(ipd.Audio(y, rate=16000, normalize=False))
            plt.plot(y)
        return y

    @staticmethod
    def read_txt(path):
        with open(path, 'r') as f:
            read_list = [line.strip() for line in f]
            return read_list

    @staticmethod
    def save_txt(path, txt):
        with open(path, 'w') as f:
            f.writelines(txt)

    @staticmethod
    def adjust_audio_length(audio, max_samples):
    if tf.shape(audio)[0] > max_samples:
        new_audio = audio[:max_samples]
    else:
        padding = max_samples - tf.shape(audio)[0]
        new_audio = tf.pad(audio, [[0, padding]])
    return new_audio