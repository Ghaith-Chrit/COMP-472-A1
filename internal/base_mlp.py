import pandas as pd
from numpy import ndarray
from .utils import format_confusion_matrix_as_str
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train(mlp: MLPClassifier, trainingData: pd.DataFrame, trainingTarget: pd.DataFrame) -> None:
	mlp.fit(trainingData, trainingTarget)

def test(mlp: MLPClassifier, testingData: pd.DataFrame) -> ndarray:
	return mlp.predict(testingData)

def train_and_test(X_train, X_test, y_train, y_test, class_names: list, result_file: str) -> dict:
	mlp: MLPClassifier = MLPClassifier(hidden_layer_sizes=(100,100,), activation="logistic", solver="sgd")

	train(mlp, X_train, y_train)
	y_pred = test(mlp, X_test)
	classification_matrix = classification_report(y_test, y_pred, labels=class_names, zero_division=0)
	dict_class_matrix = classification_report(y_test, y_pred, labels=class_names, output_dict= True, zero_division=0)
	confusion_matrix_result = confusion_matrix(y_test, y_pred, labels=class_names)

	with open(result_file, "a+") as f:
		f.write("Base-MLP Results \n")
		f.write("Classification Matrix: \n")
		f.write(classification_matrix)
		f.write("\nConfusion Matrix: \n")
		f.write(format_confusion_matrix_as_str(confusion_matrix_result, class_names))
		f.write("\n" + "-" * 60 + "\n")
	return dict_class_matrix