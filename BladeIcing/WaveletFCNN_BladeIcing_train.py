from __future__ import print_function
import numpy as np
import tensorflow as tf
import pywt
from contextlib import redirect_stdout



epochs_num = 200



def wavelet_FCNN_preprocessing_set(X, waveletLevel=3, waveletFilter='haar'):
    '''
    :param X: (sample_num, feature_num, sequence_length)
    :param waveletLevel:
    :param waveletFilter:
    :return: result (sample_num, extended_sequence_length, feature_num)
    '''
    N = X.shape[0]
    feature_dim = X.shape[1]
    length = X.shape[2]
    signal_length = []
    signal_length.append(length)

    stats = []

    extened_X = []
    extened_X.append(np.transpose(X, (0, 2, 1)))

    for i in range(N):# for each sample
        for j in range(feature_dim): # for each dim
            wavelet_list = pywt.wavedec(X[i][j], waveletFilter, level=waveletLevel)
            if i == 0 and j == 0:
                for l in range(waveletLevel):
                    current_length = len(wavelet_list[waveletLevel - l])
                    signal_length.append(current_length)
                    extened_X.append(np.zeros((N,current_length,feature_dim)))
            for l in range(waveletLevel):
                extened_X[l+1][i,:,j] = wavelet_list[waveletLevel-l]

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

    print(result.shape)
    #print(signal_length)
    #print(stats)
    return result, signal_length


def wavelet_FCNN_preprocessing_test_set(X, stats, waveletLevel=3, waveletFilter='haar'):
    assert(len(stats)==waveletLevel+1)
    N = X.shape[0]
    feature_dim = X.shape[1]

    extened_X = []
    extened_X.append(np.transpose(X,(0,2,1)))

    for i in range(N):  # for each signal
        for j in range(feature_dim):
            wavelet_list = pywt.wavedec(X[i][j], waveletFilter, level=waveletLevel)
            if i == 0:
                for l in range(waveletLevel):
                    current_length = len(wavelet_list[waveletLevel - l])
                    extened_X.append(np.zeros((N, current_length, feature_dim)))
            for l in range(waveletLevel):
                extened_X[l + 1][i,:,j] = wavelet_list[waveletLevel - l]

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


def wavelet_FCNN_model(waveletLevel=4, waveletFilter='haar'):
    # load the data
    x_train = np.load('./GoldWind/train_X.npy')
    y_train = np.load('./GoldWind/train_Y.npy')
    x_test = np.load('./GoldWind/test_X.npy')
    y_test = np.load('./GoldWind/test_Y.npy')

    # pre-processing the data
    class_num = len(np.unique(y_train))
    batch_size = int(min(x_train.shape[0]/100, 16))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (class_num - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (class_num - 1)

    y_train =tf.keras.utils.to_categorical(y_train, class_num)
    y_test = tf.keras.utils.to_categorical(y_test, class_num)

    x_train, signal_length = wavelet_FCNN_preprocessing_set(x_train, waveletLevel=waveletLevel,waveletFilter=waveletFilter)
    x_test, _ = wavelet_FCNN_preprocessing_set(x_test,waveletLevel=waveletLevel, waveletFilter=waveletFilter)

    #x_train = x_train.reshape(x_train.shape + (1,))
    #x_test = x_test.reshape(x_test.shape + (1,))
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

    model_save = tf.keras.callbacks.ModelCheckpoint('./GoldWind/wavelet_FCNN_Model.hdf5', save_best_only=True,
                                                    monitor='val_acc', mode='max')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.001)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_num,
              verbose=1, validation_data=(x_test, y_test), callbacks = [reduce_lr, model_save])



def train(level=4):
    with open('./GoldWind/wavelet_FCNN_train_level_'+str(level)+'.log','w') as log:
        with redirect_stdout(log):
            wavelet_FCNN_model(level)


def main():
    train()

if __name__ == '__main__':
    main()


