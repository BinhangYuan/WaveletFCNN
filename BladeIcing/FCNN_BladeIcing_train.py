'''
This code follows the implementation of

'''



from __future__ import print_function
import numpy as np
import tensorflow as tf
from contextlib import redirect_stdout


epochs_num = 200



def FCNN_model():
    # load the data
    x_train = np.load('./GoldWind/train_X.npy')
    y_train = np.load('./GoldWind/train_Y.npy')
    x_test = np.load('./GoldWind/test_X.npy')
    y_test = np.load('./GoldWind/test_Y.npy')

    # pre-processing the data
    class_num = len(np.unique(y_train))
    batch_size = int(min(x_train.shape[0] / 100, 16))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (class_num - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (class_num - 1)

    y_train =tf.keras.utils.to_categorical(y_train, class_num)
    y_test = tf.keras.utils.to_categorical(y_test, class_num)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()

    x_train = (x_train - x_train_mean) / (x_train_std)
    x_test = (x_test - x_train_mean) / (x_train_std)

    x = tf.keras.layers.Input(x_train.shape[1:])

    conv1 = tf.keras.layers.Conv1D(filters=128,kernel_size=8,strides=1,padding="SAME")(x)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.layers.Activation("relu")(bn1)

    conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="SAME")(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    relu2 = tf.keras.layers.Activation("relu")(bn2)

    conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding="SAME")(relu2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3)
    relu3 = tf.keras.layers.Activation("relu")(bn3)

    global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(relu3)
    softmax = tf.keras.layers.Dense(units=class_num, activation='softmax')(global_avg_pool)

    model = tf.keras.Model(inputs=x, outputs=softmax)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    model_save = tf.keras.callbacks.ModelCheckpoint('./GoldWind/baseline_FCNN_Model.hdf5', save_best_only=True,
                                                    monitor='val_acc', mode='max')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5, patience=50, min_lr=0.001)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_num,
              verbose=1, validation_data=(x_test, y_test), callbacks = [reduce_lr, model_save])



def train():
    with open('./GoldWind/baseline_FCNN_train.log','w') as log:
        with redirect_stdout(log):
            FCNN_model()


def main():
    train()

if __name__ == '__main__':
    main()

