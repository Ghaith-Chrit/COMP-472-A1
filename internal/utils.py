from numpy import ndarray

def format_confusion_matrix_as_str(confusion_matrix: ndarray, class_names: list) -> str:
	formatted_matrix = "{:<20}\t" + "\t".join(["{:<10}" for _ in range(len(class_names))])
	confusion_matrix_str = formatted_matrix.format('Predicted\Target', *class_names) + "\n"
	for i, row in enumerate(confusion_matrix.T):
		confusion_matrix_str += formatted_matrix.format(class_names[i], *row) + "\n"
	return confusion_matrix_str.strip()

def format_dict_as_str(data: dict) -> str:
	return"\n".join([f"- {key}: {value}" for key, value in data.items()]) + "\n"