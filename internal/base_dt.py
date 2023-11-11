import pandas as pd
from sklearn import tree
from numpy import ndarray
from .data_visualizations import visualize_graph
from .utils import format_confusion_matrix_as_str
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix

def train(dtc: tree.DecisionTreeClassifier, trainingData: pd.DataFrame, trainingTarget: pd.DataFrame) -> None:
	dtc.fit(trainingData, trainingTarget)

def test(dtc: tree.DecisionTreeClassifier, testingData: pd.DataFrame) -> ndarray:
	return dtc.predict(testingData)

def train_and_test(data: pd.DataFrame, target_col_name: str, graph_file: str, result_file: str, graph_max_depth: int | None = None) -> dict:
	X, y = data.drop([target_col_name], axis=1), data[target_col_name]
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	class_names: list = y.unique().astype('str').tolist()
	dtc: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier()

	train(dtc, X_train, y_train)
	visualize_graph(dtc, feature_names=X_train.columns, class_names=class_names, output=graph_file, max_depth=graph_max_depth)
	y_pred = test(dtc, X_test)
	classification_matrix = classification_report(y_test, y_pred, labels=class_names)
	dict_class_matrix = classification_report(y_test, y_pred, labels=class_names, output_dict= True)

	confusion_matrix_result = confusion_matrix(y_test, y_pred, labels=class_names)

	with open(result_file, "a+") as f:
		f.write("Base-DT Results \n")
		f.write("Classification Matrix: \n")
		f.write(classification_matrix)
		f.write("\nConfusion Matrix: \n")
		f.write(format_confusion_matrix_as_str(confusion_matrix_result, class_names))
		f.write("\n" + "-" * 60 + "\n")
	return dict_class_matrix