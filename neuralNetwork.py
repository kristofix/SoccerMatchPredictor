from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib

import numpy as np
matplotlib.use('TkAgg')  # Or use another backend that you have, like 'Qt5Agg'


def nn_model(X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]  # Get the number of features from the training data

    model = Sequential()
    model.add(Dense(n_features, input_dim=n_features, activation='relu'))  # Number of neurons = n_features
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1, mode='min')

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    # Save the model
    model.save('my_model.h5')

    import matplotlib.pyplot as plt

# Plotting on one image using subplots
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
