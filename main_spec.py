import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import json

from tensorflow.keras import layers
from tensorflow.keras import models

from utils import UtilsIO
from utils_metrics import UtilsMetrics

AUTOTUNE = tf.data.AUTOTUNE

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path, label):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  waveform = tf.cast(waveform, dtype=tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == LABELS)
  return spectrogram, label_id

def preprocess_dataset(files, labels):
  files_ds = tf.data.Dataset.from_tensor_slices((files, labels))
  output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
  return output_ds


def select_results_by_gender(df_test, x):
    df_test_x = df_test[df_test["genders"] == x]
    test_files_x, test_labels_x = UtilsIO.get_files(df_test_x)
    test_audio_x, test_labels_x = UtilsIO.get_test_set(preprocess_dataset(test_files_x, test_labels_x))
    y_true_x, y_pred_x, test_acc_x = UtilsMetrics.get_results(test_audio_x, test_labels_x)
    return test_acc_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config_filepath = args.config

    with open(config_filepath, "r") as f:
        config = json.load(f)
    
    # Load dataset    
    print("________________ Load dataset ________________")

    exp_path = f"results/{config['name']}"
    os.makedirs(exp_path, exist_ok=True)
    
    df_train = pd.read_csv(config['database']['train'])
    df_val = pd.read_csv(config['database']['val'])
    df_test = pd.read_csv(config['database']['test'])

    print("Train: ", len(df_train), "Val: ", len(df_val), "Test: ", len(df_test))

    LABELS = list(df_train["words"].unique())
    num_labels = len(LABELS)

    train_files, train_labels = UtilsIO.get_files(df_train)
    val_files, val_labels = UtilsIO.get_files(df_val)
    test_files, test_labels = UtilsIO.get_files(df_test)
    
    train_ds = preprocess_dataset(train_files, train_labels)
    val_ds = preprocess_dataset(val_files, val_labels)
    
    # Extract info model
    print("________________ Get model ________________")
    
    for spectrogram, _ in val_ds.take(1):
        input_shape = spectrogram.shape
    
    print('Input shape:', input_shape)

    norm_layer = layers.Normalization()
    norm_layer.adapt(data=val_ds.map(map_func=lambda spec, label: spec))

    shape_resize = config['model']['shape_resize']
    s1, s2 = shape_resize

    model = models.Sequential([
        layers.Input(shape=input_shape),    
        layers.Resizing(s1, s2), # Downsample the input.
        norm_layer, # Normalize.
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

    batch_size = config['model']['batch_size']
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)

    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    epochs = config['model']['epochs']
    print(f"Start train! Num Epochs:  {epochs}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
    )

    metrics = history.history
    UtilsMetrics.plot_curve(metrics, exp_path, history)

    model.save(f"{exp_path}/model.keras")
    

    print("________________ Test model ________________")

    test_ds = preprocess_dataset(test_files, test_labels)
    test_audio, test_labels = UtilsIO.get_test_set(test_ds)

    print("General results: ")
    y_true, y_pred, general_acc = UtilsMetrics.get_results(test_audio, test_labels, model)
    UtilsMetrics.conf_matrix(y_true, y_pred, exp_path, LABELS)

    name = f"{exp_path}/results_general_spec.csv"
    UtilsMetrics.save_results(df_test, test_files, test_labels, y_pred, name)
    
    print("Accuracy Female: ")
    acc_female = select_results_by_gender(df_test, "F")

    print("Accuracy Male: ")
    acc_male = select_results_by_gender(df_test, "M")

    with open(f"{exp_path}/summary_results.txt", "w") as f:
        f.write(f"General accuracy: {general_acc:.0%}\n")
        f.write(f"Female accuracy: {acc_female:.0%}\n")
        f.write(f"Male accuracy: {acc_male:.0%}\n")

    # plot the distribution of pitch in the training and test dataset
    UtilsMetrics.get_histogram(exp_path, config)
    



