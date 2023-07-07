from sklearn.preprocessing import StandardScaler, MinMaxScaler

from abc import ABC, abstractmethod

class Transformer(ABC):
    '''Classe abstrata que contém operaçoes de um preprocessador de dados. '''
    @abstractmethod
    def transform_data(self,dataset):
        pass

class StandardScalerTransformer(ABC):

    def __init__(self):
        print("Setting Up StandardScalerTransformer")
        self.name = "StandardScaler"
        self.transformer = StandardScaler()

    

class MinMaxScalerTransformer(ABC):

    def __init__(self):
        print("Setting Up MinMaxScalerTransformer")
        self.name = "MinMaxScaler"
        self.transformer = MinMaxScaler()