from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Concatenate
from keras.layers import Input
from keras import Model

def create_model(n_timesteps:int,
                 n_features:int,
                 n_outputs:int):
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def create_benchmark_model(n_timesteps:int,
                           n_features:int,
                           n_outputs:int):
    input = Input(shape=(n_timesteps,n_features))
    x = GRU(units=64)(input)
    y = GRU(units=64,go_backwards=True)(input)
    z = Concatenate()([x,y])
    z = Dense(64,activation="selu")(z)
    z = Dense(n_outputs,activation="softmax")(z)
    model = Model(inputs=input,outputs = z)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model