import h5py
import tensorflow as tf
import math
import numpy as np
from keras.utils import Sequence
import os
import time

DATA_PATH = 'radiodataset0to16/'
TRAIN_DS_PATH = DATA_PATH + 'Train/train_data.h5'  # PATH TO TRAINING SET
VAL_DS_PATH = DATA_PATH + 'Validation/val_data.h5'  # PATH TO VALIDATION SET
TEST_FOLDER_PATH = DATA_PATH + "Test/"  # PATH TO TEST FOLDER
CLASSES_PATH = DATA_PATH + 'classes-fixed.txt'


class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, filepath, batch_size=256):
        self.filepath = filepath
        self.dims = size_of_data(filepath)
        self.batch_size = batch_size
        self.indices = tf.range(self.dims)

    def shuffle(self):
        self.indices = tf.random.shuffle(self.indices)

    def shuffle_batch(self, batch_x, batch_y):
        indx = tf.range(self.batch_size, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indx)

        shuffled_x = tf.gather(batch_x, shuffled_indices)
        shuffled_y = tf.gather(batch_y, shuffled_indices)
        return shuffled_x, shuffled_y

    def __len__(self):
        return int(math.ceil(self.dims / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        inds = tf.sort(inds, axis=-1, direction='ASCENDING')
        with h5py.File(self.filepath, 'r') as hf:
            batch_x = hf['X'][inds]
            batch_y = hf['Y'][inds]

            batch_x = preprocessing_batch_data(batch_x)

        return self.shuffle_batch(batch_x, np.argmax(batch_y, axis=1))

    def on_epoch_end(self):
        # self.indices = tf.random.shuffle(self.indices)
        self.shuffle()


def size_of_data(filepath):
    with h5py.File(filepath, 'r') as hf:
        return hf['X'].shape[0]


@tf.function
def preprocessing_batch_data(batched_data):
    """preprocessing data function"""
    # preprocessing
    # 1 - RMS norm
    abs_sqrd = batched_data[:, :, 0] ** 2 + batched_data[:, :, 1] ** 2
    rms = tf.sqrt(tf.reduce_mean(abs_sqrd, axis=1, keepdims=True))
    rms = tf.expand_dims(rms, axis=-1)  # (batch_size, 1, 1)
    batch_norm = batched_data / rms  # (batch_size, 1024, 2)

    # 2 - Transposing
    batch_transpose = tf.transpose(batch_norm, perm=[0, 2, 1])  # (batch_size, 2, 1024)

    # 3 - Expanding dims
    batch_dims = tf.expand_dims(batch_transpose, axis=-1)  # (batch_size, 2, 1024, 1)
    return batch_dims


def generate_datasets(batch_size=32):
    train_datagen = Generator(TRAIN_DS_PATH, batch_size)
    train_datagen.shuffle()
    val_datagen = Generator(VAL_DS_PATH, batch_size)
    return train_datagen, val_datagen


def get_datasets_number_of_samples():
    train_len = size_of_data(TRAIN_DS_PATH)
    val_len = size_of_data(VAL_DS_PATH)
    return train_len, val_len


def class_names():
    return np.loadtxt(CLASSES_PATH, dtype=str)


def evaluate_model_on_test_set(model):
    score = {}
    time_logs = []
    for root, dir_names, file_names in os.walk(TEST_FOLDER_PATH):
        if file_names:
            current_db = os.path.basename(root)
            current_file = root + '/' + file_names[0]
            print(f'db: {current_db}')
            print('file: ' + current_file)

            with h5py.File(current_file, 'r') as hf:
                x = hf['X'][:]
                y = hf['Y'][:]

            x = preprocessing_batch_data(x)
            y = np.argmax(y, axis=1)
            current_score = model.evaluate(x, y)[1]  # select accuracy
            
            #x_batch = preprocessing_batch_data(x)
            #aftest_labels = np.argmax(y, axis=1)
            #prediction_digits = []
            #for i, x_test in enumerate(x_batch):
                
            #    x_test = np.expand_dims(x_test, axis=0).astype(np.float32)

            #    start_time=time.time()
            #    output = model.predict(x_test)
            #    end_time=time.time()

            #    duration= end_time-start_time
            #    time_logs.append(duration)
            #    digit = np.argmax(output[0])
            #    prediction_digits.append(digit)
           # prediction_digits = np.array(prediction_digits)
           # accuracy = (prediction_digits == test_labels).mean()

            # start_time=time.time()
            # y_logs = model.predict(x)
            # end_time=time.time()

            # y_pred = np.argmax(y_logs, axis=1)
            # current_score = np.mean(y == y_pred)
            # score[int(current_db)] = current_score

            # duration= end_time-start_time
            # time_logs.append(duration)

            # score[int(current_db)] = accuracy
            score[int(current_db)] = current_score
            print(f'score: {current_score}')
    score = {i: score[i] for i in np.sort(list(score.keys()))}
    # return score, time_logs
    return score

def evaluate_model_on_test_set_tfl_interpreter(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    score = {}
    time_logs = []
    for root, dir_names, file_names in os.walk(TEST_FOLDER_PATH):
        if file_names:
            current_db = os.path.basename(root)
            current_file = root + '/' + file_names[0]
            print(f'db: {current_db}')
            print('file: ' + current_file)
            prediction_digits = []
            with h5py.File(current_file, 'r') as hf:
                x = hf['X'][:]
                y = hf['Y'][:]

            x_batch = preprocessing_batch_data(x)
            test_labels = np.argmax(y, axis=1)
            for i, x_test in enumerate(x_batch):
                
                x_test = np.expand_dims(x_test, axis=0).astype(np.float32)
                
                start_time=time.time()
                interpreter.set_tensor(input_index, x_test)
                
                #run inference
                interpreter.invoke()

                #post-processing: remove batch dimension and find the digit with highest
                #probability
                output = interpreter.tensor(output_index)
                end_time=time.time()
                duration= end_time-start_time
                time_logs.append(duration)
                digit = np.argmax(output()[0])
                prediction_digits.append(digit)
            prediction_digits = np.array(prediction_digits)
            accuracy = (prediction_digits == test_labels).mean()

            score[int(current_db)] = accuracy
            print(f'score: {accuracy}')
    score = {i: score[i] for i in np.sort(list(score.keys()))}
    return score, time_logs