import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class Plotter(ABC):
    """Classe abstrata que contém operacoes de um Plotter."""

    @abstractmethod
    def plot(self,y_test,predictions):
        """Método que plota o grafico.""" 
        pass

class MatPlotter(Plotter):
    def __init__(self):
        self.__xlabel = "Samples"
        self.__ylabel = "y and predictions"
        self.__title = "y and predictions comparison"
        self.__legend = "upper left"

    def plot(self,y_test,predictions):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(predictions,"-b", Label="y_hat")
        #TODO alertar rafael do problema desse codigo
        plt.plot(y_test,"-r", Label="y")
        x1,x2,y1,y2 = plt.axis()
        x1 = 0
        x2 = 50
        plt.axis((x1,x2,y1,y2))
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.title(self.__title)
        plt.legend(Loc=self.__legend)
        plt.show()