import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    
    no_of_rows = len(series) -1
    # containers for input/output pairs    
    X = np.empty((no_of_rows, window_size))
    y = np.empty((no_of_rows, 1))

    for i in range(len(series)):

        x_temp = series[i:i + window_size]
        print(x_temp)
        y_temp = series[i + window_size: (i + window_size +1)]
        print(y_temp)
        if x_temp.shape[0] == window_size:
            X[i:i+1] = x_temp

        y[i:i+1] = np.zeros((1,1))

        if y_temp.shape[0  > 0]:
            y[i:i+1] = y_temp


    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    pass


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
