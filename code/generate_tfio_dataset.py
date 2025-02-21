import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import h5py

DATA_PATH = 'radiodataset0to16/'
TRAIN_DS_PATH = DATA_PATH + 'Train/train_data.h5'  # PATH TO TRAINING SET
VAL_DS_PATH = DATA_PATH + 'Validation/val_data.h5'  # PATH TO VALIDATION SET


@tf.function
def preprocessing_x_data(batched_data):
    """preprocessing data function"""
    # preprocessing
    # (1024, 2)
    abs_sqrd = batched_data[:, 0] ** 2 + batched_data[:, 1] ** 2 
    rms = tf.sqrt(tf.reduce_mean(abs_sqrd, axis=0, keepdims=True))
    # rms = tf.expand_dims(rms, axis=-1)  # (batch_size, 1, 1)
    batch_norm = batched_data / rms  # (1024, 2)

    # 2 - Transposing
    batch_transpose = tf.transpose(batch_norm, perm=[1, 0])  # (2, 1024)

    # 3 - Expanding dims
    batch_dims = tf.expand_dims(batch_transpose, axis=-1)  # (2, 1024, 1)
    return batch_dims

def preprocessing_dataset(dataset):
    return dataset.map(lambda x, y: (preprocessing_x_data(x), tf.argmax(y)))

def size_of_data(filepath):
    with h5py.File(filepath, 'r') as hf:
        return hf['X'].shape[0]


def generate_datasets(batch_size=32, buffer_size_ratio=4):
    x_train_ds = tfio.IODataset.from_hdf5(TRAIN_DS_PATH, ['/X'], num_parallel_reads=5)
    y_train_ds = tfio.IODataset.from_hdf5(TRAIN_DS_PATH, ['/Y'], num_parallel_reads=5)
    
    x_val_ds = tfio.IODataset.from_hdf5(VAL_DS_PATH, ['/X'], num_parallel_reads=4)
    y_val_ds = tfio.IODataset.from_hdf5(VAL_DS_PATH, ['/Y'], num_parallel_reads=4)
    
    train_ds = tf.data.Dataset.zip((x_train_ds, y_train_ds))
    val_ds = tf.data.Dataset.zip((x_val_ds, y_val_ds))
    
    train_len = size_of_data(TRAIN_DS_PATH)
    val_len = size_of_data(VAL_DS_PATH)
    
    train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(train_len))
    val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(val_len))
    
    train_ds = preprocessing_dataset(train_ds)
    val_ds = preprocessing_dataset(val_ds)
    
    BUFFERSIZE = int(train_len/buffer_size_ratio)
    
    train_ds = train_ds.shuffle(BUFFERSIZE).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE) #.cache() might help but i dont have enough memory
    val_ds = val_ds.batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
