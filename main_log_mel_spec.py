import os
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd
import json

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from utils import UtilsIO

AUTOTUNE = tf.data.AUTOTUNE

SAMPLE_RATE = 16000
MAX_LENGTH = 19200

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def get_files(df):
    files = list(df["filepaths"])
    labels = list(df["words"])
    return files, labels

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path, label):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_log_mel_spectrogram(waveform):
    waveform = UtilsIO.adjust_audio_length(waveform, MAX_LENGTH)
    waveform = tf.cast(waveform, dtype=tf.float32)
    stfts = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 40
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SAMPLE_RATE, 
        lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return log_mel_spectrograms[..., tf.newaxis] # Return (Frames, Bins, 1)

def get_log_mel_spectrogram_and_label_id(audio, label):
  log_mel_spec = get_log_mel_spectrogram(audio)
  label_id = tf.argmax(label == LABELS)
  return log_mel_spec, label_id

def preprocess_dataset(files, labels):
  files_ds = tf.data.Dataset.from_tensor_slices((files, labels))
  output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(map_func=get_log_mel_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
  return output_ds

def get_test_set(test_ds):
    test_audio = []
    test_labels = []
    
    for audio, label in test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())
    
    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    return test_audio, test_labels

def get_results(test_audio, test_labels):
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels
    
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    return y_true, y_pred    

def plot_curve(metrics, path):
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.savefig(f"{path}/curve.png")

def conf_matrix(y_true, y_pred, path):
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig(f"{path}/confusion_matrix.png")

def select_results_by_gender(df_test, x):
    df_test_x = df_test[df_test["genders"] == x]
    test_files_x, test_labels_x = get_files(df_test_x)
    test_audio_x, test_labels_x = get_test_set(preprocess_dataset(test_files_x, test_labels_x))
    y_true_x, y_pred_x = get_results(test_audio_x, test_labels_x)

def get_histogram(exp_path, type_, input_path):
    df = pd.read_csv(input_path)
    plt.figure(figsize=(10, 4))
    df["f0"].hist(bins=100, density=True)
    df["f0"].plot.kde()
    plt.title(f"Histogram - {type_}")
    plt.savefig(f"{exp_path}/Histogram_{type_}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config_filepath = args.config

    with open(config_filepath, "r") as f:
        config = json.load(f)

    MAX_LENGTH = config['params']['max_length']
    
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

    train_files, train_labels = get_files(df_train)
    val_files, val_labels = get_files(df_val)
    test_files, test_labels = get_files(df_test)

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
    plot_curve(metrics, exp_path)

    model.save(f"{exp_path}/model.keras")

    print("________________ Test model ________________")

    test_ds = preprocess_dataset(test_files, test_labels)
    test_audio, test_labels = get_test_set(test_ds)

    print("General results: ")
    y_true, y_pred = get_results(test_audio, test_labels)
    conf_matrix(y_true, y_pred, exp_path)
    
    print("Accuracy Female: ")
    select_results_by_gender(df_test, "F")

    print("Accuracy Male: ")
    select_results_by_gender(df_test, "M")

    # plot the distribution of pitch in the training and test dataset
    get_histogram(exp_path, "train", config['database']['train'])
    get_histogram(exp_path, "test", config['database']['test'])
    



