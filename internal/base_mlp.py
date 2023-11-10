import pandas as pd
from numpy import ndarray
from .utils import format_confusion_matrix_as_str
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train(mlp: MLPClassifier, trainingData: pd.DataFrame, trainingTarget: pd.DataFrame) -> None:
	mlp.fit(trainingData, trainingTarget)

def test(mlp: MLPClassifier, testingData: pd.DataFrame) -> ndarray:
	return mlp.predict(testingData)

def train_and_test(data: pd.DataFrame, target_col_name: str, result_file: str) -> None:
	X, y = data.drop([target_col_name], axis=1), data[target_col_name]
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	class_names: list = y.unique().astype('str').tolist()
	mlp: MLPClassifier = MLPClassifier(hidden_layer_sizes=(100,100), activation="logistic", solver="sgd")

	train(mlp, X_train, y_train)
	y_pred = test(mlp, X_test)
	classification_matrix = classification_report(y_test, y_pred, labels=class_names)
	confusion_matrix_result = confusion_matrix(y_test, y_pred, labels=class_names)

	with open(result_file, "a+") as f:
		f.write("Base-MLP Results \n")
		f.write("Classification Matrix: \n")
		f.write(classification_matrix)
		f.write("\nConfusion Matrix: \n")
		f.write(format_confusion_matrix_as_str(confusion_matrix_result, class_names))
		f.write("\n" + "-" * 60 + "\n")