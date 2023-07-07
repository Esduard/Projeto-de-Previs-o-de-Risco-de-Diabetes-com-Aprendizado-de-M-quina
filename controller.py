

from data_loader import DataLoader
from evaluator import Evaluator
from model import Model
from plotter import Plotter
from pre_processor import PreProcessor
from transformer import Transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class Controller():

    def __init__(self,dataLoader: DataLoader,preProcessor: PreProcessor,
                 models: Model,transformers: Transformer,evaluator : Evaluator, plotter: Plotter):
        self.__dataloader = dataLoader
        self.__preProcessor = preProcessor
        self.__models = models
        self.__transformers = transformers
        self.__evaluator = evaluator
        self.__plotter = plotter


    def __processModel(self,pipelines: list, model: Model, X_train, X_test, y_train, y_test):
        scoring = 'accuracy'
        num_particoes = 10
        for name, estimator in pipelines:
            if name.split('-')[0] != "Voting":
                print('\nGridSearching {}'.format(name))
                kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, 
                                        random_state=self.__preProcessor.seed)
                grid_search = GridSearchCV(estimator=estimator, param_grid=model.grid, scoring=scoring, cv=kfold)
                grid_search.fit(X_train, y_train)
                print('\n{} - Melhor: {} usando {}'.format(name, grid_search.best_score_, grid_search.best_params_))
                
                estimator.set_params(**grid_search.best_params_)
            
            #Model Training
            print('\nTraining for {}'.format(name))
            estimator.fit(X_train, y_train)
            
            #Model Predict
            predictions = estimator.predict(X_test)
            #Evaluation
            print('\nEstimation for {}'.format(name))
            self.__evaluator.evaluate(y_test,predictions,name)

    
    def run(self):
        #Load Dataset
        dataset = self.__dataloader.dataset()
        #Preprocessing
        x_train, x_test, y_train, y_test = self.__preProcessor.preprocess(dataset)

        for model in self.__models:
            model_tuple = (model.name,model.model)

            pipelines = [('{}-orig'.format(model.name), Pipeline(steps=[model_tuple]) )]
            for transformer in self.__transformers:
                model_tuple = (model.name,model.model)
                transformer_tuple = (transformer.name,transformer.transformer)
                pipelines.append(('{}-{}'.format(model.name, transformer.name), 
                              Pipeline(steps=[transformer_tuple,model_tuple]) ))
            self.__processModel(pipelines, model, x_train, x_test, y_train, y_test)
        self.__evaluator.show_best_results()
        