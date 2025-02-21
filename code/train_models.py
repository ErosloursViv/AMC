from build_model import build_sbcnn_model, build_ibcnn_model
# from generate_datasets import generate_datasets, get_datasets_number_of_samples
from generate_tfio_dataset import generate_datasets
import tensorflow as tf
import math
from auxiliar import get_time_stamp, create_dir
import os

SAVE_PATH = 'saved_models'

def create_sbcnn_model(optimizer=tf.keras.optimizers.AdamW()):
    sbcnn = build_sbcnn_model()
    sbcnn.compile(
        # Optimizer
        optimizer=optimizer,
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return sbcnn


def create_ibcnn_model(sbcnn, optimizer=tf.keras.optimizers.AdamW()):
    ibcnn = build_ibcnn_model(sbcnn, trainable=True)
    ibcnn.compile(
        optimizer=optimizer,  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return ibcnn


def train_model(epochs, batch_size, *model, optimizer=None):
    if model == ():
        model = create_sbcnn_model(optimizer) if optimizer else create_sbcnn_model()
        name = 'sbcnn'
    else:
        model = create_ibcnn_model(*model, optimizer) if optimizer else create_ibcnn_model(*model)
        name = 'ibcnn'

    train_datagen, val_datagen = generate_datasets(batch_size=batch_size)
    # train_len, val_len = get_datasets_number_of_samples()

    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=name + '_best.keras',
            save_weights_only=False,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.CSVLogger(name + f'{get_time_stamp()}log.csv', separator=",", append=True)
    ]

    model_history = model.fit(train_datagen,
                              # steps_per_epoch=int(math.ceil(train_len // batch_size)),
                              validation_data=val_datagen,
                              # validation_steps=int(math.ceil(val_len // batch_size)),
                              epochs=epochs,
                              callbacks=callbacks)

    create_dir(SAVE_PATH)
    # save_path = f"{SAVE_PATH}/{name}{get_time_stamp()}.weights.h5"
    save_path = f"{SAVE_PATH}/{name}.weights.h5"
    save_dir = os.path.dirname(save_path)
    model.save_weights(save_path)

    return model_history, model

def train_optimized_model(model, epochs, batch_size, callbacks):
    train_datagen, val_datagen = generate_datasets(batch_size=batch_size)
    # train_len, val_len = get_datasets_number_of_samples()

    model_history = model.fit(train_datagen,
                              # steps_per_epoch=int(math.ceil(train_len // batch_size)),
                              validation_data=val_datagen,
                              # validation_steps=int(math.ceil(val_len // batch_size)),
                              epochs=epochs,
                              callbacks=callbacks)
    return model_history

def load_from_model(filepath):
    model = tf.keras.models.load_model(filepath)
    return model


def load_from_weights(filepath):
    sbcnn = create_sbcnn_model()
    if 'sbcnn' in filepath.lower():
        sbcnn.load_weights(filepath)
        return sbcnn
    else:
        ibcnn = create_ibcnn_model(sbcnn)
        ibcnn.load_weights(filepath)
        return ibcnn


def clone_from_filepath(filepath,**clone_function):
    model_to_clone = load_from_model(filepath)
    clone = tf.keras.models.clone_model(
    model_to_clone,
    clone_function
    )
    clone.load_weights(model_to_clone.weights)
    return clone