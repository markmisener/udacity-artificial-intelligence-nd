import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras

import re
import string

# TODO: fill out the function below that transforms the input series 

def window_transform_series(series, window_size):
    """
    Run a sliding window along an input series to create associated input/output pairs
    """
    # containers for input/output pairs
    X = []
    y = []
    
    # loop over the input series, appending an array of items from the series of length window_size to X
    for i in range(0, (len(series) - window_size)):
        X.append(series[i:(i + window_size)])
        
    # output values
    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    """
    Use Keras to quickly build a two hidden layer RNN of the following specifications:
    - Layer 1 uses an LSTM module with 5 hidden units (of input_shape = (window_size,1))
    - Layer 2 uses a fully connected module with one unit
    - The 'mean_squared_error' loss should be used (remember: we are performing regression here)
    """
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1, activation='linear'))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    """
    Replace non-ascii lowercase and common punctuation from a given text
    """
    # create a list of characters we want to keep
    punctuation = ['!', ',', '.', ':', ';', '?']
    keep_characters = punctuation + list(string.ascii_lowercase)
    
    # create a list of the characters of the text if they are not in the keep_characters list
    remove_characters = [x for x in text if x not in keep_characters]
    # create a regular expression pattern of the characters we want to remove
    regex = re.compile('|'.join(map(re.escape, remove_characters)))
    
    # substitute an empty character for each of the characters we want to remove
    text = re.sub(regex, ' ', text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    """
    Create a function that runs a sliding window along the input text and creates associated input/output pairs. A skeleton function has been provided for you. Note that this function should input a) the text b) the window size and c) the step size, and return the input/output sequences. Note: the return items should be lists - not numpy arrays.
    """
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # loop over the input series, appending an array of items from the series of length window_size to inputs and outputs
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    
    return inputs, outputs

# TODO build the required RNN model: 
def build_part2_RNN(window_size, num_chars):
    """
    Use Keras to quickly build a single hidden layer RNN - where our hidden layer consists of LSTM modules.
    - Layer 1 should be an LSTM module with 200 hidden units of input_shape:
        - (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    - Layer 2 should be a linear module, fully connected, of len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    - Layer 3 should be a softmax activation ( since we are solving a multiclass classification)
    - Use the categorical_crossentropy loss
    """
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
