from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
#from tensorflow.keras import regularizers
from wandb.keras import WandbCallback
import wandb
from config import epochs, patience
import tensorflow as tf
import numpy as np

# I'm using sparse_categorical_crossentropy loss function, but below I'm preparing place for custom loss already
def custom_loss(y_true, y_pred):
    return tf.keras.metrics.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1, ignore_class=None)

def nn_model(X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]

    model = Sequential()
    model.add(Dense(n_features, input_dim=n_features, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=1, mode='min')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stop, WandbCallback()])

    model.save('nn_model.keras')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Log Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show(block=False)

    wandb.log({"Training Metrics": [wandb.Image(fig)]})


