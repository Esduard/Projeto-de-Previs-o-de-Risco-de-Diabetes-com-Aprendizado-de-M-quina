from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest # para a Seleção Univariada

from abc import ABC, abstractmethod

class PreProcessor(ABC):
    '''Classe abstrata que contém operaçoes de um preprocessador de dados. '''
    @abstractmethod
    def preprocess(self,dataset):
        pass

class SelectKBestPreProcessor(PreProcessor):
    def __init__(self,columns,k =4, seed =0, test_percentage=0.30):
        print("Setting Up SelectKBestPreProcessor k={}".format(k))
        self.features = columns[:-1]
        self.label = columns[-1]
        self.seed = seed
        self.TEST_PERCENTAGE = test_percentage
        self.k = k
    
    def __separate_feature_and_label(self,dataset):
        dataframe = dataset
        dataframe.sort_index(inplace=True)
        x = dataframe[self.features].values
        y = dataframe[self.label].values
        return x,y
    
    def __selectKBest(self,X,y):
        best_var = SelectKBest(score_func=f_classif, k=self.k)
        fit = best_var.fit(X, y)
        return fit.transform(X)
        

    def preprocess(self,dataset):
        X,y = self.__separate_feature_and_label(dataset)
        X = self.__selectKBest(X,y)
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=self.TEST_PERCENTAGE, random_state=self.seed)
        return x_train, x_test, y_train, y_test