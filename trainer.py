import numpy as np

class Trainer():
    def __init__(self,model,
                 verbose:int,
                 epochs:int,
                 batch_size:int
                 ):
        self.model = model
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size

    def train_on_new_data(self,x:np.array,
                          y:np.array):
        self.model.fit(x,y,
          epochs=self.epochs,
          batch_size=self.batch_size,
          verbose=self.verbose)