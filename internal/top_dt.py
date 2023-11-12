import pandas as pd
from sklearn import tree
from numpy import ndarray
from .data_visualizations import visualize_graph
from .utils import format_confusion_matrix_as_str, format_dict_as_str
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def train(clf: GridSearchCV, trainingData: pd.DataFrame, trainingTarget: pd.DataFrame) -> None:
	clf.fit(trainingData, trainingTarget)

def test(clf: GridSearchCV, testingData: pd.DataFrame) -> ndarray:
	return clf.predict(testingData)

def train_and_test(X_train, X_test, y_train, y_test, class_names: list, graph_file: str, result_file: str, graph_max_depth: int | None = None) -> dict:
	dtc: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier()
	clf: GridSearchCV = GridSearchCV(dtc, param_grid={'criterion': ['gini','entropy'], 'max_depth': [5, 20, None], 'min_samples_split': [5,8,10]})
	
	train(clf, X_train, y_train)
	visualize_graph(clf.best_estimator_, feature_names=X_train.columns, class_names=class_names, output=graph_file, max_depth=graph_max_depth)
	y_pred = test(clf, X_test)
	classification_matrix = classification_report(y_test, y_pred, target_names=class_names)
	dict_class_matrix = classification_report(y_test, y_pred, labels=class_names, output_dict= True)
	confusion_matrix_result = confusion_matrix(y_test, y_pred, labels=class_names)

	with open(result_file, "a+") as f:
		f.write("Top-DT Results \n")
		f.write("Hyper-Parameters: \n")
		f.write(format_dict_as_str(clf.best_params_))
		f.write("\nClassification Matrix: \n")
		f.write(classification_matrix)
		f.write("\nConfusion Matrix: \n")
		f.write(format_confusion_matrix_as_str(confusion_matrix_result, class_names))
		f.write("\n" + "-" * 60 + "\n")
	return dict_class_matrix