import numpy as np
from controller import Controller
from data_loader import DataLoaderFromLocal
from evaluator import ClassificationEvaluator
from model import AdaBoostClassifierModel, BaggingClassifierModel, CARTClassifier, ExtraTreesClassifierModel, GradientBoostingClassifierModel, KNNClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, RandomForestClassifierModel, VotingClassifierModel
from pre_processor import SelectKBestPreProcessor
from transformer import MinMaxScalerTransformer, StandardScalerTransformer


dataset_name = 'diabetes.csv'

columns = ['Gravidezes', 'Glucose', 'Pressão sanguínea', 'Espessura da pele', 'Insulina', 'IMC', 'Pedigree de diabetes', 'Idade', 'Resultado']

dataLoader = DataLoaderFromLocal('./livroescd-main/', dataset_name, columns)

seed = 7
np.random.seed(seed)

preProcessor = SelectKBestPreProcessor(seed = seed, test_percentage=0.30, columns=columns, k=4)

evaluator = ClassificationEvaluator(target_names=['0', '1'])

transformers = [
    StandardScalerTransformer(),
    MinMaxScalerTransformer()
    ]


models = [
    VotingClassifierModel(),
    LogisticRegressionClassifier(),
    KNNClassifier(),
    CARTClassifier(),
    NaiveBayesClassifier(),
    BaggingClassifierModel(),
    RandomForestClassifierModel(),
    ExtraTreesClassifierModel(),
    AdaBoostClassifierModel(),
    GradientBoostingClassifierModel(),
    ]

controller = Controller(dataLoader,preProcessor,
                 models,transformers,evaluator, None)

controller.run()