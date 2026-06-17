import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm


def organize_paths(filepath, gain):
    out_folder = os.path.dirname(filepath).replace("GPS_cmd_16k_renamed", "GPS_cmd_16k_renamed_mix_white")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.basename(filepath).replace(".wav", f"_white_noise_{gain}dB.wav")
    out = f"{out_folder}/{out_path}"
    return out


def apply_white_noise(speech_path, output_path, target_snr_db):
    sr = 16000
    speech = librosa.load(speech_path, sr=sr)[0]
    white_noise = np.random.normal(loc=0, scale=1, size=len(speech))
    power_speech = np.mean(speech ** 2)
    power_noise = np.mean(white_noise ** 2)    
    if power_noise == 0:
        return speech
    # Formula derived from: SNR_dB = 10 * log10(Power_clean / Power_noise_scaled)
    target_snr_linear = 10 ** (target_snr_db / 10.0)
    scaling_factor = np.sqrt(power_speech / (power_noise * target_snr_linear))
    speech_mixed = speech + (scaling_factor * white_noise)
    speech_mixed_norm = librosa.util.normalize(speech_mixed)
    sf.write(output_path, speech_mixed_norm, sr)


if '__main__' == __name__:
    
    #input_csv = "../lists/GPS_cmd_pyin_setup_1_test.csv"
    #input_csv = "../lists/GPS_cmd_pyin_setup_1_val.csv"
    input_csv = "../lists/GPS_cmd_pyin_setup_1_train.csv"
    df = pd.read_csv(input_csv)
    gains = [5, 10, 20]

    mixed_filepaths = []
    for filepath in tqdm(df['filepaths']):
        gain = np.random.choice(gains)
        output_filepath = organize_paths(filepath, gain)
        mixed_filepaths.append(output_filepath)
        apply_white_noise(filepath, output_filepath, gain)

    df["filepath_wn"] = mixed_filepaths
    name = input_csv.replace(".csv", "_white_noise.csv")
    df.to_csv(name, index=False)
        
        
    
