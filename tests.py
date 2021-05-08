import numpy as np
import tensorflow as tf
from tensorflow import keras

import csv

from tools import *
from speaker_recognition import *
#valid_audio_paths, valid_labels, paths_and_labels_to_dataset, add_noise, noises, audio_to_fft, class_names

loaded_model = keras.models.load_model("model.h5")
SAMPLES_TO_DISPLAY = 100

def read_from_csv(filename):
    list = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            list.append(row)
    return list

if __name__ == "__main__":
    valid_audio_paths = read_from_csv("valid_audio_paths.csv")
    valid_labels = read_from_csv("valid_labels.csv")

    valid_audio_paths = valid_audio_paths[0]
    valid_labels = valid_labels[0]
    valid_labels = list(map(int, valid_labels))

    noise_paths = noise_prepare()
    noises = create_noises(noise_paths)

    audio_paths, labels, class_names = get_paths_labels()
    test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y))

    #print(test_ds)

    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = audio_to_fft(audios)
        # Predict
        y_pred = loaded_model.predict(ffts)
        # Take random samples
        rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        #tf.cast(labels, tf.uint8)
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

        for index in range(SAMPLES_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            #print(labels[index])
            print(
                "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                    "[92m" if labels[index] == y_pred[index] else "[91m",
                    class_names[labels[index]],
                    "[92m" if labels[index] == y_pred[index] else "[91m",
                    class_names[y_pred[index]],
                )
            )
            #display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))
