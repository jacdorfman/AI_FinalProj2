import pandas as pd
import numpy as np
import os

from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import to_categorical


raw_data_path = 'raw_data/'

def create_model():
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels und pixel data
        output: image and label array """
    
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['Emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'Pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
        
    return image_array, image_label


def data_to_tf_data(df):
    image_array, image_label = prepare_data(df)
    images = image_array.reshape((image_array.shape[0], 48, 48, 1))
    images = images.astype('float32')/255
    labels = to_categorical(image_label)
    return images, labels

if __name__ == "__main__":
    train = pd.read_csv(raw_data_path+'initial_training_data.csv')
    train_images, train_labels = data_to_tf_data(train)

    val = pd.read_csv(raw_data_path+'validation_test_data.csv')
    val_images, val_labels = data_to_tf_data(val)


    model = create_model()
    class_weight = dict(zip(range(0, 7), (((train['Emotion'].value_counts()).sort_index())/len(train['Emotion'])).tolist()))
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        class_weight = class_weight,
                        epochs=12,
                        batch_size=64)

    df = pd.read_csv(raw_data_path+'test_data.csv')
    test_images, test_labels = data_to_tf_data(df)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
