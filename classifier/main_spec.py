import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

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

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == LABELS)
  return spectrogram, label_id

def preprocess_dataset(files, labels):
  files_ds = tf.data.Dataset.from_tensor_slices((files, labels))
  output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
  return output_ds

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


if __name__ == '__main__':
    # Load dataset
    print("Load dataset...")
    abs_path = "../pitch/GPS_cmd_16k_renamed_pyin_avg_by_word_user"
    # TODO: adjust the save path
    exp_path = "results/"
    df_train = pd.read_csv(f"{abs_path}_train.csv")
    df_val = pd.read_csv(f"{abs_path}_val.csv")
    df_test = pd.read_csv(f"{abs_path}_test.csv")

    LABELS = list(df_train["words"].unique())
    num_labels = len(LABELS)

    train_files, train_labels = get_files(df_train)
    val_files, val_labels = get_files(df_val)
    test_files, test_labels = get_files(df_test)

    train_ds = preprocess_dataset(train_files, train_labels)
    val_ds = preprocess_dataset(val_files, val_labels)
    test_ds = preprocess_dataset(test_files, test_labels)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Extract info model
    print("Get model...")
    
    for spectrogram, _ in val_spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    
    print('Input shape:', input_shape)

    norm_layer = layers.Normalization()
    norm_layer.adapt(data=val_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),    
        layers.Resizing(32, 32), # Downsample the input.
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

    EPOCHS = 10
    print(f"Start train! Num Epochs:  {EPOCHS}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plot_curve(metrics, exp_path)

    print("Test model...")
    test_audio = []
    test_labels = []
    
    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())
    
    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels
    
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy - general: {test_acc:.0%}')

    conf_matrix(y_true, y_pred, exp_path)

    # TODO: select performance by gender
    



