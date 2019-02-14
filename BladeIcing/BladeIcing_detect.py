from __future__ import print_function
import numpy as np
import tensorflow as tf
import pywt


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
    return result, signal_length



def predict_data_set(model_path = './GoldWind/wavelet_FCNN_Model.hdf5', X_data_path='./GoldWind/test_X.npy',Y_data_path=None):
    model = tf.keras.models.load_model(filepath=model_path)
    if model_path == './GoldWind/wavelet_FCNN_Model.hdf5':
        test_set_X, _ = wavelet_FCNN_preprocessing_set(np.load(X_data_path), waveletLevel=4)
    elif model_path == './GoldWind/baseline_FCNN_Model.hdf5':
        test_set_X =np.load(X_data_path)
    predict_Y = model.predict(test_set_X)
    if Y_data_path:
        test_set_Y = np.load(Y_data_path)
        print("Test accuracy:",np.mean(test_set_Y==np.argmax(predict_Y,axis=1)))
    return np.argmax(predict_Y,axis=1)


def moving_average_helper(array, window_size=5):
    weights = np.repeat(1.0, window_size) / window_size
    moving_averaged_array = np.convolve(array, weights, 'valid')
    array[window_size//2:-(window_size//2)+1] = moving_averaged_array
    return array


def moving_vote(X_data_path, Y_data_path, model_path='./GoldWind/wavelet_FCNN_Model.hdf5', threshold=0.5):
    Y_data = np.load(Y_data_path)
    N_normal = np.sum(Y_data == 0)
    N_fault = np.sum(Y_data == 1)
    init_predict_Y = predict_data_set(model_path, X_data_path)
    print(init_predict_Y)
    TP = np.sum((init_predict_Y == 1) * (Y_data == 1))
    FN = np.sum((init_predict_Y == 0) * (Y_data == 1))
    FP = np.sum((init_predict_Y == 1) * (Y_data == 0))
    TN = np.sum((init_predict_Y == 0) * (Y_data == 0))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision + recall)
    print("Initial TP:",TP, " TN:",TN, " FP:", FP, " FN:",FN, " precision:", precision, " recall:", recall, " score:", f1)
    moving_averaged_predict_Y = moving_average_helper(init_predict_Y.astype(float),window_size=32)
    print(moving_averaged_predict_Y)
    final_predict_Y = moving_averaged_predict_Y > threshold
    TP = np.sum((final_predict_Y==1)*(Y_data==1))
    FN = np.sum((final_predict_Y==0)*(Y_data==1))
    FP = np.sum((final_predict_Y==1)*(Y_data==0))
    TN = np.sum((final_predict_Y==0)*(Y_data==0))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy =  (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("Voted TP:",TP, " TN:",TN, " FP:", FP, " FN:",FN, " accuracy: ", accuracy, " precision:", precision, " recall:", recall, " score:", f1)


def straw_man(X_data_path, Y_data_path, model_path='./GoldWind/wavelet_FCNN_Model.hdf5'):
    Y_data = np.load(Y_data_path)[:-4]
    Y_data = np.mean(Y_data.reshape(-1,32),axis=1) >= 0.5
    init_predict_Y = predict_data_set(model_path, X_data_path)[:-4:32]
    TP = np.sum((init_predict_Y == 1) * (Y_data == 1))
    FN = np.sum((init_predict_Y == 0) * (Y_data == 1))
    FP = np.sum((init_predict_Y == 1) * (Y_data == 0))
    TN = np.sum((init_predict_Y == 0) * (Y_data == 0))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print("Strawman TP:", TP, " TN:", TN, " FP:", FP, " FN:", FN, " accuracy: ", accuracy, " precision:", precision, " recall:", recall, " score:", f1)



def main():
    print("FCNN:")
    moving_vote(X_data_path='./GoldWind/test_sequence_X.npy',
                Y_data_path='./GoldWind/test_sequence_Y.npy',
                model_path='./GoldWind/baseline_FCNN_Model.hdf5', threshold=0.3)
    print("WaveletFCNN Strawman:")
    straw_man(X_data_path='./GoldWind/test_sequence_X.npy',
              Y_data_path='./GoldWind/test_sequence_Y.npy')
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9]
    for t in thresholds:
        print("WaveletFCNN moving vote, Current threshold:", t)
        moving_vote(X_data_path='./GoldWind/test_sequence_X.npy',
                    Y_data_path='./GoldWind/test_sequence_Y.npy',
                    model_path='./GoldWind/wavelet_FCNN_Model.hdf5', threshold=t)


if __name__ == '__main__':
    main()