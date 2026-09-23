import numpy as np
import IPython.display as ipd
import glob
import librosa
import matplotlib.pyplot as plt


class MyAuxCode():
    def __init__(self):
        pass

    def plot_waveform(self, audio, sr):
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(audio) / sr, num=len(audio)), audio)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

  
    def play_audio(self, audio, sr, normalize=False):
        return ipd.display(ipd.Audio(audio, rate=sr, normalize=normalize))

    @staticmethod
    def load_audio(file_path, sr=16000):
        audio = librosa.load(file_path, sr=sr)[0]
        return audio

    def display_audio(self, file_path, sr=16000, plot=True, normalize=False):
        audio = self.load_audio(file_path, sr)
        self.play_audio(audio, sr, normalize)
        if plot:
            self.plot_waveform(audio, sr)
        return audio