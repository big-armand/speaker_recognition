import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio

import csv

from tools import *

### NOISE
def noise_prepare():
    # Get the list of all noise files
    noise_paths = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]
    print(
        "Found {} files belonging to {} directories".format(
            len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
        )
    )
    return noise_paths

    """
    Resample all noise samples to 16000 Hz
    """

    command = (
        "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
        "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
        "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
        "$file | grep sample_rate | cut -f2 -d=`; "
        "if [ $sample_rate -ne 16000 ]; then "
        "ffmpeg -hide_banner -loglevel panic -y "
        "-i $file -ar 16000 temp.wav; "
        "mv temp.wav $file; "
        "fi; done; done"
    )
    os.system(command)

# Split noise into chunks of 16,000 steps each
def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None

def create_noises(noise_paths):
    noises = []
    for path in noise_paths:
        sample = load_noise_sample(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    print(
        "{} noise files were split into {} noise samples where each is {} sec. long".format(
            len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
        )
    )
    return noises

"""
## Dataset generation
"""

def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Get the list of audio file paths along with their corresponding labels
def get_paths_labels():
    class_names = os.listdir(DATASET_AUDIO_PATH)
    print("Our class names: {}".format(class_names,))

    audio_paths = []
    labels = []
    for label, name in enumerate(class_names):
        print("Processing speaker {}".format(name,))
        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [
            os.path.join(dir_path, filepath)
            for filepath in os.listdir(dir_path)
            if filepath.endswith(".wav")
        ]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)

    print(
        "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
    )
    return audio_paths, labels, class_names

def shuffle_paths_labels(audio_paths, labels):
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)

def split(audio_paths, labels):
    # Split into training and validation
    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
    train_audio_paths = audio_paths[:-num_val_samples]
    train_labels = labels[:-num_val_samples]

    print("Using {} files for validation.".format(num_val_samples))
    valid_audio_paths = audio_paths[-num_val_samples:]
    valid_labels = labels[-num_val_samples:]

    # Create 2 datasets, one for training and the other for validation
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )

    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)
    return train_ds, valid_ds, valid_audio_paths, valid_labels


def setup_training_set(train_ds, noises):
    # Add noise to the training set
    train_ds = train_ds.map(
        lambda x, y: (add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds

def setup_validation_set(valid_ds):
    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return valid_ds

"""
## Model Definition
"""


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

"""
We get ~ 98% validation accuracy.
"""
def write_to_csv(filename, iterable):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(iterable)

if __name__ == "__main__":
    noise_paths = noise_prepare()
    noises = create_noises(noise_paths)
    audio_paths, labels, class_names = get_paths_labels()
    shuffle_paths_labels(audio_paths, labels)
    train_ds, valid_ds, valid_audio_paths, valid_labels = split(audio_paths, labels)
    write_to_csv("valid_audio_paths.csv", valid_audio_paths)
    write_to_csv("valid_labels.csv", valid_labels)

    print("SWAG")
    train_ds = setup_training_set(train_ds, noises)
    valid_ds = setup_validation_set(valid_ds)

    model = build_model((SAMPLING_RATE // 2, 1), len(class_names))
    model.summary()
    model.compile(
        optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model_save_filename = "model.h5"

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="val_accuracy", save_best_only=True
    )
    #TRAINING
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=valid_ds,
        callbacks=[earlystopping_cb, mdlcheckpoint_cb],
    )
    print(model.evaluate(valid_ds))
