import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing

import crepe

print("Number of cpu available: ", multiprocessing.cpu_count())

CORES = int(multiprocessing.cpu_count()) - 1

import sys
sys.path.append("..")
from utils import UtilsIO
utils_io = UtilsIO()

def get_pitch(path, thr_prob):
    audio = utils_io.load_audio(path)
    time, f0, confidence, activation = crepe.predict(audio, 16000, model_capacity="medium", viterbi=True)

    i_prob = np.argwhere(confidence > thr_prob)
    f0_prob = f0[i_prob]
    f0_mean = round(np.mean(f0_prob), 2)
    f0_std = round(np.std(f0_prob), 2)

    #print("pitch mean: ", f0_mean, "pitch std: ", f0_std)
    return f0_prob, f0_mean, f0_std

def main(filename):
    thr_prob = 0.8
    f0_prob, f0_mean, f0_std = get_pitch(filename, thr_prob)
    outs = [f0_prob, f0_mean, f0_std]
    return outs

if "__main__" == __name__:
    path = "../GPS_cmd_16k_renamed_jpynb.list"
    list_gps = utils_io.read_txt(path)

    list_gps = list_gps

    print("List size: ", len(list_gps))

    print("Generate dataframe...")
    genders = []
    users = []
    words = []

    for path in tqdm(list_gps):
        # input()
        user = os.path.basename(path).split("_")[1]
        word = os.path.basename(path).split("_")[0]
        gender = user[0]
        genders.append(gender)
        users.append(user)
        words.append(word)

    dict_ = {
        "filepaths": list_gps,
        "users": users,
        "genders": genders,
        "words": words
    }

    df = pd.DataFrame(dict_)

    print("Get pitch...")
    outs_f0 = process_map(main, list_gps, chunksize=10)

    print("Organize outputs...")
    f0_probs = []
    f0_stds = []
    f0_means = []
    for out_f0 in tqdm(outs_f0):
        f0_probs.append(out_f0[0])
        f0_means.append(out_f0[1])
        f0_stds.append(out_f0[2])


    df["f0_prob"] = f0_probs
    df["f0_mean"] = f0_means
    df["f0_std"] = f0_stds

    df.to_csv("GPS_cmd_16k_renamed_jpynb_crepe.csv", index=False)
    print("save file: GPS_cmd_16k_renamed_crepe.csv")