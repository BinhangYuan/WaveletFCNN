from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import pywt
import argparse

'''
This code follows the implementation in 
https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py
'''


def read_UCR_dataset(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y


def wavelet_FCNN_preprocessing_training_set(X, wavelet_level=3, wavelet_filter='haar'):
    N = X.shape[0]
    length = X.shape[1]
    signal_length = []
    signal_length.append(length)
    stats = []
    extened_X = []
    extened_X.append(X)
    for i in range(N):# for each signal
        wavelet_list = pywt.wavedec(X[i], wavelet_filter, level=wavelet_level)
        if i == 0:
            for l in range(wavelet_level):
                current_length = len(wavelet_list[wavelet_level - l])
                signal_length.append(current_length)
                extened_X.append(np.zeros((N,current_length)))
        for l in range(wavelet_level):
            extened_X[l+1][i] = wavelet_list[wavelet_level - l]
    result = None
    first = True
    for mat in extened_X:
        mat_mean = mat.mean()
        mat_std = mat.std()
        mat = (mat-mat_mean)/(mat_std)
        stats.append((mat_mean,mat_std))
        if first:
            result = mat
            first = False
        else:
            result = np.concatenate((result, mat), axis=1)
    #print(result.shape)
    return result, signal_length, stats


def wavelet_FCNN_preprocessing_test_set(X, stats, wavelet_level=3, wavelet_filter='haar'):
    assert(len(stats) == wavelet_level + 1)
    N = X.shape[0]
    extened_X = []
    extened_X.append(X)
    for i in range(N):  # for each signal
        wavelet_list = pywt.wavedec(X[i], wavelet_filter, level=wavelet_level)
        if i == 0:
            for l in range(wavelet_level):
                current_length = len(wavelet_list[wavelet_level - l])
                extened_X.append(np.zeros((N, current_length)))
        for l in range(wavelet_level):
            extened_X[l + 1][i] = wavelet_list[wavelet_level - l]
    result = None
    first = True
    for i in range(len(extened_X)):
        mat = extened_X[i]
        mat_mean = stats[i][0]
        mat_std = stats[i][1]
        mat = (mat - mat_mean) / (mat_std)
        if first:
            result = mat
            first = False
        else:
            result = np.concatenate((result, mat), axis=1)
    return result


def wavelet_FCNN_model(dataset, epochs_num, wavelet_level=4, wavelet_filter='haar'):
    # load the data
    x_train, y_train = read_UCR_dataset('./UCR_TS_Archive_2015/' + dataset + '/' + dataset + '_TRAIN')
    x_test, y_test = read_UCR_dataset('./UCR_TS_Archive_2015/' + dataset + '/' + dataset + '_TEST')

    # pre-processing the data
    class_num = len(np.unique(y_train))
    batch_size = int(min(x_train.shape[0] / 10, 16))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (class_num - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (class_num - 1)

    y_train =tf.keras.utils.to_categorical(y_train, class_num)
    y_test = tf.keras.utils.to_categorical(y_test, class_num)

    x_train, signal_length, stats = wavelet_FCNN_preprocessing_training_set(x_train, wavelet_level=wavelet_level, wavelet_filter=wavelet_filter)
    x_test = wavelet_FCNN_preprocessing_test_set(x_test, stats, wavelet_level=wavelet_level, wavelet_filter=wavelet_filter)

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    print(x_train.shape)

    x = tf.keras.layers.Input(x_train.shape[1:])
    print(x.shape)

    left = 0
    right = 0
    waveletOutput = []
    total_length = x.shape[1]
    for i in range(len(signal_length)):
        length = signal_length[i]
        right = int(total_length-left-length)
        x_crop = tf.keras.layers.Cropping1D((left, right))(x)
        left += length

        print(x_crop.shape)

        conv1 = tf.keras.layers.Conv1D(filters=128,kernel_size=8,strides=1,padding="SAME")(x_crop)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        relu1 = tf.keras.layers.Activation("relu")(bn1)

        print(relu1.shape)

        conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="SAME")(relu1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        relu2 = tf.keras.layers.Activation("relu")(bn2)

        conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding="SAME")(relu2)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        relu3 = tf.keras.layers.Activation("relu")(bn3)

        global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(relu3)
        waveletOutput.append(global_avg_pool)

    concatenate = tf.keras.layers.Concatenate(axis=1)(waveletOutput)

    softmax = tf.keras.layers.Dense(units=class_num, activation='softmax')(concatenate)

    model = tf.keras.Model(inputs=x, outputs=softmax)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5, patience=50, min_lr=0.0001)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_num,
              verbose=1, validation_data=(x_test, y_test), callbacks = [reduce_lr])
    # Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])



def main():
    parser = argparse.ArgumentParser(description='PyTorch DNN on google speech dataset (model parallelism distributed)')
    parser.add_argument('--dataset', type=str, default='50words', metavar='S',
                        help='which dataset to train')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--wavelet-level', type=int, default=4, metavar='N',
                        help='wavelet level for training (default: 4)')

    args = parser.parse_args()
    wavelet_FCNN_model(args.dataset, args.epochs, args.wavelet_level);


if __name__ == '__main__':
    main()