
#===================================#
#                                   #
#          The Boss Battle          #
#                                   #
#          Timothy Fischer          #
#             g19F7919              #
#                                   #
#===================================#

# solid 60/80

#===================================#

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

seed = 4242

#==================================================================

#   Data setup functions

#==================================================================

def get_diabetes():
    dataset = pd.read_csv('datasets/diabetes_prediction_dataset.csv')
    categorical = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
    numerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', ]
    return dataset, categorical, numerical, False

def get_breast_cancer():
    dataset = load_breast_cancer(as_frame=True)
    numerical = dataset.feature_names
    return dataset, [], numerical, True

def get_digits():
    dataset = load_digits(as_frame=True)
    numerical = dataset.feature_names
    return dataset, [], numerical, True

def get_iris():
    dataset = load_iris(as_frame=True)
    numerical = dataset.feature_names
    return dataset, [], numerical, True

def get_wine():
    dataset = load_wine(as_frame=True)
    numerical = dataset.feature_names
    return dataset, [], numerical, True

#=================================================================

#   Helper functions

#=================================================================

def get_dataset(dataset):
    match dataset:
        case 'diabetes':
            return get_diabetes()
        case 'breast_cancer':
            return get_breast_cancer()
        case 'digits':
            return get_digits()
        case 'iris':
            return get_iris()
        case 'wine':
            return get_wine()
        case default:
            EXIT(f"Unknown Dataset: \'{dataset}\'")

def fill_missing(x):
    if x.isnull().sum().sum() > 0:
        x = x.fillna(x.median())
    return x

def split_data(dataset, sk):
    if (sk):
        X = dataset.data
        y = dataset.target
    else:
        X = dataset.iloc[:,0:-1]
        y = dataset.iloc[:,-1]
    return train_test_split(fill_missing(X), y, train_size=0.7, random_state=seed)

def get_scaler(scaler):
    match scaler:
        case 'min_max':
            return MinMaxScaler()
        case 'standard':
            return StandardScaler()
        case default:
            EXIT(f"Unknown Scaler: \'{scaler}\'")

def  get_selector(selector):
    match selector:
        case 'kbest':
            return SelectKBest()
        case 'pca':
            return PCA()
        case default:
            EXIT(f"Unknown Selector: \'{selector}\'")

def display_eda(selector): #WIP
    match selector:
        case 'pca':
            # sns.scatterplot(data= ,x=, y=, hue=)
            return
        case default:
            return

#=================================================================

#   Model selection

#=================================================================

def get_model(model):
    match model:
        case 'knn':
            parameters = {
                'model__n_neighbors' : [1, 3, 5]
            }
            return 'K Nearest Neighbours', KNeighborsClassifier(), parameters
        
        case 'logistic':
            parameters = {
                'model__penalty' : ['l1', 'l2']
            }
            return 'Logistic Regression', LogisticRegression(), parameters
        
        case 'dt':
            parameters = {
                'model__criterion' : ['gini', 'entropy']
            }
            return 'Decision Tree', DecisionTreeClassifier(), parameters
        
        case 'rf':
            parameters = {
                'model__criterion' : ['gini', 'entropy']
            }
            return 'Random Forest', RandomForestClassifier(), parameters
        
        case 'aboost':
            parameters = {}
            return 'AdaBoost', AdaBoostClassifier(), parameters
        
        case 'gboost':
            parameters = {}
            return 'GradientBoost', GradientBoostingClassifier(), parameters
        
        case 'svm':
            parameters = { 
                'model__C' : [0.01, 1, 10],
                'model__gamma' : [1, 10, 100],
                'model__kernel' : ['rbf', 'poly', 'sigmoid'] 
            }
            return 'Support Vector machine', SVC(), parameters
        
        case 'mlp':
            parameters = {
                'model__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                'model__alpha': [0.0001, 0.05]
            }
            return 'Multi-layer Perceptron', MLPClassifier(learning_rate='adaptive'), parameters
        
        case default:
            EXIT(f"Unknown Model: \'{model}\'")

#=======================================================================

# The Pipeline

#======================================================================

def generate_pipeline(scaler, selector, model, numercal_cols, categorical_cols):
    numerical_pipeline = Pipeline(steps=[
        ('scale', scaler)
    ])

    categorical_pipeline = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical_pipeline', numerical_pipeline, numercal_cols),
        ('nategorical_pipeline', categorical_pipeline, categorical_cols)
    ], remainder='drop', n_jobs=-1)

    model_pipeline = Pipeline(steps=[
        ('column_transformer', column_transformer),
        ('select', selector),
        ('model', model)
    ])
    return model_pipeline

#===============================================================

# The Battle Field

#===============================================================

def battle(dataset, scaler, selector, model, metric, eda):
    dataset, categorical_cols, numercal_cols, sk = get_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(dataset, sk)
    scaler = get_scaler(scaler)
    selector = get_selector(selector)
    name, model, parameters = get_model(model)
    pipeline = generate_pipeline(scaler, selector, model, numercal_cols, categorical_cols)

    if (selector == 'kbest'):
        s_params = {
            'select__k' : [3, 5, 10]
        }
        parameters = parameters.update(s_params)

    grid = GridSearchCV(pipeline, parameters, scoring=(metric), cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)
        
    y_pred = grid.predict(X_test)
    result = classification_report(y_test, y_pred)
    print(f"Classification Report for {name} using Best Parameters:\n\n{result}")

    result = [
        round(accuracy_score(y_test, y_pred), 2),
        round(precision_score(y_test, y_pred, average='weighted'), 2),
        round(recall_score(y_test, y_pred, average='weighted'), 2),
        round(f1_score(y_test, y_pred, average='macro'), 2),
        name
    ]
    return result

def show_results(results, metric):
    plt.figure(figsize=(11,6))
    plt.title(f'{metric} score for each Classifier')
    ax = sns.barplot(data=results, x=results['Classifier'], y=results[metric])
    for i, j in enumerate(results[metric]):
        ax.text(i, j/2, str(j), ha='center', color='white')
    ax.spines[['top', 'right']].set_visible(False)
    plt.xlabel("Classifier")
    plt.ylabel("scores")
    plt.show()

#===============================================================

# Main Method

#===============================================================

def main():
    parser = argparse.ArgumentParser(description='Throw in a dataset, choose you\'re classifier and watch the chaos ensue. Please refer to the readme file to see all options!')
    parser.add_argument('-d', '--dataset', help='Select the dataset you want to work with.')
    parser.add_argument('-sc', '--scaler', nargs='?', const='min_max', help='Select the scaler you want to use (default: min_max).')
    parser.add_argument('-se', '--selector', nargs='?', const='kbest', help='Select the type of feature selection you want to use (default: kbest).')
    parser.add_argument('-c', '--classifier', nargs='+', help='Select the machine learning classifier you want to use (accepts multiple inputs).')
    parser.add_argument('-m', '--metric', nargs='?', const='accuracy', help='Select the metric you want to use for cross validation (default: accuracy).')
    parser.add_argument('-eda', action='store_true', help='(Optional) Display EDA for the chosen dataset.')

    args = parser.parse_args()
    dataset = args.dataset
    scaler = args.scaler
    selector = args.selector
    metric = args.metric
    classifiers = args.classifier
    eda = args.eda

    final_results = []
    for c in classifiers:
        result = battle(dataset, scaler, selector, c, metric, eda)
        final_results.append(result)
    dataframe = pd.DataFrame(data=final_results, columns=['accuracy', 'precision', 'recall' ,'f1_macro', 'Classifier'])
    show_results(dataframe, metric)

def EXIT(error):
    print(f"\nERROR: {error}\n")
    exit()

if (__name__ == '__main__'):
    main()