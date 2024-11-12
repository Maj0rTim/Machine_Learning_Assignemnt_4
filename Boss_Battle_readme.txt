The Boss Battle
Timothy Fischer
g19f7919

self mark: 60/80

Description:

	Throw in a dataset, choose you're classifier and watch the chaos ensue. 

Example Commands:

	The_Boss_Battle.py -d 'wine' -sc 'min_max' -se 'pca' -m 'f1_macro' -c 'KNN' 'DT' 'SVM'
	The_Boss_Battle.py -d 'diabetes' -sc 'min_max' -se 'kbest' -m 'accuracy' -c 'rf' 'gboost' 'mlp'


Description of Arguments:

-d --dataset
	
	Select the scaler you want to use (default: min_max)
	options: 'diabetes', 'breast_cancer', 'digits', 'iris', 'wine'

-sc --scaler

	Select the scaler you want to use (default: min_max)
	options: 'min_max', 'standard'

-ce --selector

	Select the type of feature selection you want to use (default: kbest)
	options: 'kbest', 'pca'

-c --classifier

	Select the machine learning classifier you want to use (accepts multiple inputs)
	options: 'logistic', 'knn', 'dt', 'rf', 'aboost', 'gboost', 'svm', 'mlp'

-m --metric

	Select the metric you want to use for cross validation (default: accuracy)
	options: 'accuracy', 'precision', 'recall', 'f1_macro'

-eda

	(Optional) Display EDA for the chosen dataset.Does not take in a value. 
	This is still a work in progress :(