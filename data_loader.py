from abc import ABC, abstractmethod
import os

import pandas as pd

class DataLoader(ABC):
    '''Classe abstrata que contém operacées de um carregador de dados.'''

    @abstractmethod
    def dataset(self):
        '''Método que retorna um dataset.'''
        pass
    
class DataLoaderFromLocal(DataLoader):
    def __init__(self,data_directory,file_name, columns):
        print("Loading Data Locally")
        full_path = os.path.join(data_directory,file_name)
        dataset = pd.read_csv(full_path, names=columns, skiprows=1, delimiter=',')
        self.__dataset = dataset

    def dataset(self):
        return self.__dataset