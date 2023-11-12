import pandas as pd
from numpy import ndarray
from .utils import format_confusion_matrix_as_str, format_dict_as_str
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train(clf: GridSearchCV, trainingData: pd.DataFrame, trainingTarget: pd.DataFrame) -> None:
	clf.fit(trainingData, trainingTarget)

def test(clf: GridSearchCV, testingData: pd.DataFrame) -> ndarray:
	return clf.predict(testingData)

def train_and_test(X_train, X_test, y_train, y_test, class_names: list, result_file: str) -> dict:
	mlp: MLPClassifier = MLPClassifier(max_iter=700)
	clf: GridSearchCV = GridSearchCV(mlp, param_grid={'activation': ['logistic','tanh','relu'], 'hidden_layer_sizes': [(30,50,),(10,20,10,)], 'solver': ['sgd','adam']})

	train(clf, X_train, y_train)
	y_pred = test(clf, X_test)
	classification_matrix = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
	dict_class_matrix = classification_report(y_test, y_pred, labels=class_names, output_dict= True, zero_division=0)
	confusion_matrix_result = confusion_matrix(y_test, y_pred, labels=class_names)

	with open(result_file, "a+") as f:
		f.write("Top-MLP Results \n")
		f.write("Hyper-Parameters: \n")
		f.write(format_dict_as_str(clf.best_params_))
		f.write("\nClassification Matrix: \n")
		f.write(classification_matrix)
		f.write("\nConfusion Matrix: \n")
		f.write(format_confusion_matrix_as_str(confusion_matrix_result, class_names))
		f.write("\n" + "-" * 60 + "\n")
	return dict_class_matrix