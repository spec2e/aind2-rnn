import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    
    no_of_rows = len(series) - window_size
    # containers for input/output pairs    
    X = np.empty((no_of_rows, window_size))
    y = np.empty((no_of_rows, 1))
    
    for i in range(len(series)):
        x_temp = series[i:i + window_size]
        y_temp = series[i + window_size: (i + window_size +1)]
      
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
    model = Sequential()
    model.add(LSTM(units=5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    def is_ascii_letter(ch):
        ascii_value = ord(ch)
        return ascii_value > 96 and ascii_value < 123
    
    text = list(text)
    for i, char in enumerate(text):
        if not char in punctuation and not is_ascii_letter(char):
            text[i] = ' '

    text = ''.join(text)
    
    return text

def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    max_range = len(text) - window_size
    for i in range(0, max_range, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    
    return inputs,outputs


# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):

    model = Sequential()
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model