import os
import json
import librosa
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm
from scipy.io import wavfile


def organize_paths(filepath, value, effect_name, suffix=""):
    out_folder = os.path.dirname(filepath).replace("GPS_cmd_16k_renamed", f"GPS_cmd_16k_renamed_mix_{effect_name}")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.basename(filepath).replace(".wav", f"_{value}{suffix}_{effect_name}.wav")
    out = f"{out_folder}/{out_path}"
    return out

def apply_echo(file_path, output_path, delay_seconds=0.3, attenuation=0.5):
    sample_rate, data = wavfile.read(file_path)
    signal = data.astype(np.float32)
    echos = int(delay_seconds * sample_rate)
    
    output_signal = np.zeros(len(signal) + echos, dtype=np.float32)
    output_signal[:len(signal)] += signal
    output_signal[echos:echos + len(signal)] += signal * attenuation
    
    # Normalize the audio back to 16-bit boundaries to prevent distortion
    output_signal_int = np.clip(output_signal, -32768, 32767).astype(np.int16)
    #return output_signal
    output_signal_norm = librosa.util.normalize(output_signal)
    wavfile.write(output_path, sample_rate, output_signal_norm)


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

def normalize_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak == 0:          # avoid division by zero on pure silence
        return audio
    return audio / peak
        

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config_filepath = args.config

    with open(config_filepath, "r") as f:
        config = json.load(f)

    
    input_csv = config['input_csv']
    effects = config['effects']

    print(effects)

    df = pd.read_csv(input_csv)

    for effect_name in effects.keys():
        if effect_name == "reverb" and effects[effect_name]['apply']:
            values_rev = effects[effect_name]['values']
            print("Effect: ", effect_name)
        
            mixed_filepaths = []
            for filepath in tqdm(df['filepaths']):
                value = np.random.choice(values_rev)
                output_filepath = organize_paths(filepath, value, effect_name)
                apply_echo(filepath, output_filepath, delay_seconds=value, attenuation=0.2)
                mixed_filepaths.append(output_filepath)

            df_new = df.copy()
            df_new["filepaths"] = mixed_filepaths
            name = input_csv.replace(".csv", f"_{effect_name}.csv")
            df_new.to_csv(name, index=False)
            print("Save csv: ", name)
            print("-"*80)

        if effect_name == "white_noise" and effects[effect_name]['apply']:
            values_wn = effects[effect_name]['values']
            print("Effect: ", effect_name)
        
            mixed_filepaths = []
            for filepath in tqdm(df['filepaths']):
                value = np.random.choice(values_wn)
                output_filepath = organize_paths(filepath, value, effect_name)
                apply_white_noise(filepath, output_filepath, value)
                mixed_filepaths.append(output_filepath)

            df_new = df.copy()
            df_new["filepaths"] = mixed_filepaths
            name = input_csv.replace(".csv", f"_{effect_name}.csv")
            df_new.to_csv(name, index=False)
            print("Save csv: ", name)
            print("-"*80)

        
    
        # effect_name = "reverb"
        # values_rev = [0.06, 0.09]
        
        # # effect_name = "white_noise"
        # # values_wn = [10, 30]
        
        # print("Effect: ", effect_name)
    
        # mixed_filepaths = []
        # for filepath in tqdm(df['filepaths']):
            
        #     if effect_name == "reverb":
        #         value = np.random.choice(values_rev)
        #         output_filepath = organize_paths(filepath, value, effect_name)
        #         apply_echo(filepath, output_filepath, delay_seconds=value, attenuation=0.2)
        #     elif effect_name == "white_noise":
        #         value = np.random.choice(values_wn)
        #         output_filepath = organize_paths(filepath, value, effect_name)
        #         apply_white_noise(filepath, output_filepath, value)
    
        #     mixed_filepaths.append(output_filepath)
    
        # df["mixed_filepath"] = mixed_filepaths
        # name = input_csv.replace(".csv", f"_{effect_name}.csv")
        # df.to_csv(name, index=False)
        # print("Save csv: ", name)
        # print("-"*80)
        
    
