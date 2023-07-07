from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


from abc import ABC, abstractmethod

class Evaluator(ABC):
    """Classe abstrata que contém opreações de um Avaliador"""

    @abstractmethod
    def evaluate(self,y_test,predictions):
        pass

 
class RegressionEvaluator(Evaluator):

    def calculate_mean_squared_error(self,y_test,predictions):
        return mean_squared_error(y_test, predictions)


    def calculate_r2_score(self,y_test, predictions):
        return r2_score(y_test, predictions)

    def evaluate(self,y_test,predictions):
        print("Mean Squared Error:", self.calculate_mean_squared_error(y_test,predictions))
        print("R2 score:", self.calculate_r2_score(y_test,predictions))

class ClassificationEvaluator(Evaluator):

    def __init__(self, target_names=None):
        print("Setting Up ClassificationEvaluator")
        self.target_names = target_names
        self.best_results = { 'precision' : {
            '0' : ('none',-1),
            '1' : ('none',-1),
            'accuracy' : ('none',-1),
            'macro avg' : ('none',-1),
            'weighted avg' : ('none',-1),
        },
        'recall' : {
            '0' : ('none',-1),
            '1' : ('none',-1),
            'accuracy' : ('none',-1),
            'macro avg' : ('none',-1),
            'weighted avg' : ('none',-1),
        },
        'f1-score' : {
            '0' : ('none',-1),
            '1' : ('none',-1),
            'accuracy' : ('none',-1),
            'macro avg' : ('none',-1),
            'weighted avg' : ('none',-1),
        },
        'support' : {
            '0' : ('none',-1),
            '1' : ('none',-1),
            'accuracy' : ('none',-1),
            'macro avg' : ('none',-1),
            'weighted avg' : ('none',-1),
        },

        }
        self.all_results = {}
    
    def evaluate(self,y_test,predictions,name):
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        print("\nClassification Report:")
        cr_dict = classification_report(y_test, predictions, target_names=self.target_names, output_dict=True)
        df = pd.DataFrame(cr_dict).transpose()
        print(df)
        for column in df:
            for index, _ in df.iterrows():
                value = df.loc[index, column]
                if value > self.best_results[column][index][1]:
                    self.best_results[column][index] = (name,value)
        self.all_results[name] = df

    def show_best_results(self):
        print("\nBest results sheet")
        df = pd.DataFrame(self.best_results)
        print(df)

        dic_sum_diff = {}

        print("\nResults graded by standard deviation (smaller is better)")
        for name, results in self.all_results.items():
            sum_diff = 0.0
            for column in results:
                for index, _ in results.iterrows():
                    value = results.loc[index, column]
                    best = self.best_results[column][index][1]
                    sum_diff = sum_diff + (best - value)
            dic_sum_diff[name] = sum_diff
        
        sum_diff_series = pd.Series(dic_sum_diff).sort_values(ascending=True)

        print(sum_diff_series)
        
