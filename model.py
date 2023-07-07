from sklearn.linear_model import LinearRegression

from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

class Model(ABC):
    """Classe abstrata que contém operacoes de um Modelo."""

    def fit(self, x_train, y_train):
        """Método para realizacao de treinamento."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """Método para realizacao de predicoes."""
        return self.model.predict(x_test)

class LogisticRegressionClassifier(Model):
    def __init__(self, max_iter=200):
        print("Setting Up LogisticRegressionClassifier")
        self.name = "LR"
        self.model = LogisticRegression(max_iter=max_iter)
    grid = {
        'LR__max_iter': [100, 200, 300, 1000, 2000, 3000],
        'LR__C': [0.1, 1.0, 10.0]
    }

class KNNClassifier(Model):
    def __init__(self):
        print("Setting Up KNNClassifier")
        self.name = "KNN"
        self.model = KNeighborsClassifier()
    grid = {
        'KNN__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
        'KNN__weights': ['uniform', 'distance'],
        'KNN__metric': ["euclidean", "manhattan", "minkowski"],
    }

class CARTClassifier(Model):
    def __init__(self):
        print("Setting Up CARTClassifier")
        self.name = "CART"
        self.model = DecisionTreeClassifier()
    grid = {
        'CART__max_depth': [None, 5, 10],
        'CART__min_samples_split': [2, 5, 10]
    }

class NaiveBayesClassifier(Model):
    def __init__(self):
        print("Setting Up NaiveBayesClassifier")
        self.name = "NB"
        self.model = GaussianNB()
    grid = {}

class SVMClassifier(Model):
    def __init__(self):
        print("Setting Up SVMClassifier")
        self.name = "SVM"
        self.model = SVC()
    grid = {
        'SVM__C': [0.1, 1.0, 10.0],
        'SVM__kernel': ['linear', 'rbf']
    }

class BaggingClassifierModel(Model):
    def __init__(self):
        print("Setting Up BaggingClassifierModel")
        self.name = "Bag"
        self.model = BaggingClassifier()
    grid = {
        'Bag__n_estimators': [50, 100, 200],
        'Bag__max_samples': [0.5, 0.75, 1.0]
    }

class RandomForestClassifierModel(Model):
    def __init__(self):
        print("Setting Up RandomForestClassifierModel")
        self.name = "RF"
        self.model = RandomForestClassifier()
    grid = {
        'RF__n_estimators': [50, 100, 200],
        'RF__max_depth': [None, 5, 10]
    }

class ExtraTreesClassifierModel(Model):
    def __init__(self):
        print("Setting Up ExtraTreesClassifierModel")
        self.name = "ET"
        self.model = ExtraTreesClassifier()
    grid = {
        'ET__n_estimators': [50, 100, 200],
        'ET__max_depth': [None, 5, 10]
    }

class AdaBoostClassifierModel(Model):
    def __init__(self):
        print("Setting Up AdaBoostClassifierModel")
        self.name = "Ada"
        self.model = AdaBoostClassifier()
    grid = {
        'Ada__n_estimators': [50, 100, 200],
        'Ada__learning_rate': [0.1, 0.5, 1.0]
    }

class GradientBoostingClassifierModel(Model):
    def __init__(self):
        print("Setting Up GradientBoostingClassifierModel")
        self.name = "GB"
        self.model = GradientBoostingClassifier()
    grid = {
        'GB__n_estimators': [50, 100, 200],
        'GB__learning_rate': [0.1, 0.5, 1.0]
    }

class VotingClassifierModel(Model):
    grid = {}
    def __init__(self):
        print("Setting Up VotingClassifierModel")
        self.name = "Voting"
        bases_classes = [
                        ('LR', LogisticRegressionClassifier()),
                        ('CART', CARTClassifier()),
                        ('SVM', SVMClassifier()),
                        ('Bag', BaggingClassifierModel()),
                        ('RF', RandomForestClassifierModel()),
                        ('ET', ExtraTreesClassifierModel()),
                        ('Ada', AdaBoostClassifierModel()),
                        ('GB', GradientBoostingClassifierModel()),
                      ]
        bases = []
        for name, model in bases_classes:
            bases.append((name,model.model))
        self.model = VotingClassifier(estimators=bases)
    
        
        