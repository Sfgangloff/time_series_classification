from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Concatenate
from keras.layers import Input,Bidirectional,GlobalMaxPooling1D
from keras import Model

def create_model(n_timesteps:int,
                 n_features:int,
                 n_outputs:int,
                 model_type:int=0):
    if model_type==0:
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
    elif model_type==1:
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    elif model_type==2:
        input = Input(shape=(n_timesteps,n_features))
        x1 = Bidirectional(LSTM(units=512, return_sequences=True))(input)
        
        l1 = Bidirectional(LSTM(units=384, return_sequences=True))(x1)
        l2 = Bidirectional(LSTM(units=384, return_sequences=True))(input)

        c1 = Concatenate(axis=2)([l1,l2])

        l3 = Bidirectional(LSTM(units=256, return_sequences=True))(c1)
        l4 = Bidirectional(LSTM(units=256, return_sequences=True))(l2)
        
        c2 = Concatenate(axis=2)([l3,l4])
        
        l6 = GlobalMaxPooling1D()(c2)
        l7 = Dense(units=128, activation='selu')(l6)
        l8 = Dropout(0.05)(l7)
        
        output = Dense(n_outputs, activation='sigmoid')(l8)

        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    
    return model