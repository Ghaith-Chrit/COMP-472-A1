from typing import Callable
from numpy import ndarray
import statistics

def format_confusion_matrix_as_str(confusion_matrix: ndarray, class_names: list) -> str:
	formatted_matrix = "{:<20}\t" + "\t".join(["{:<10}" for _ in range(len(class_names))])
	confusion_matrix_str = formatted_matrix.format('Predicted\Target', *class_names) + "\n"
	for i, row in enumerate(confusion_matrix.T):
		confusion_matrix_str += formatted_matrix.format(class_names[i], *row) + "\n"
	return confusion_matrix_str.strip()

def format_dict_as_str(data: dict) -> str:
	return"\n".join([f"- {key}: {value}" for key, value in data.items()]) + "\n"

def run_model_times(model: Callable[..., dict], model_description: str, **kwargs) -> list[dict]:
	results = []
	for i in range(5):
		print(f"Running {model_description}: Run number {i + 1}")
		results.append(model(**kwargs))
	print()
	return results

def calculate_average_and_variance(data: list[dict])-> dict:
	accuracy = []
	macro = []
	weighted = []

	for i in range(len(data)):
		accuracy.append(data[i]["accuracy"])
		macro.append(data[i]["macro avg"]["f1-score"])
		weighted.append(data[i]["weighted avg"]["f1-score"])
		
	avg_acc = statistics.mean(accuracy)
	var_acc = statistics.variance(accuracy)
	avg_macro = statistics.mean(macro)
	var_macro = statistics.variance(macro)
	avg_weighted = statistics.mean(weighted)
	var_weighted = statistics.variance(weighted)

	dict_avg_var = {
		"accuracy": avg_acc,
		"var_accuracy": var_acc,
		"macro":avg_macro,
		"var_macro": var_macro,
		"weighted": avg_weighted,
		"var_weighted": var_weighted
	}

	return dict_avg_var

def write_average_and_variance_to_file(file_name, dict_to_print, type_classifier):
	with open(file_name, "a+") as f:
		if type_classifier == "BaseDT":
			f.write("The Averages and variance over 5 iterations of Base-DT: \n")
		elif type_classifier == "TopDT":
			f.write("The Averages and variance over 5 iterations of Top-DT: \n")
		elif type_classifier == "BaseMLP":
			f.write("The Averages and variance over 5 iterations of Base-MLP: \n")
		elif type_classifier == "TopMLP":
			f.write("The Averages and variance over 5 iterations of Top-MLP: \n")

		f.write("- Average Accuracy: ")
		f.write(str(dict_to_print["accuracy"]))
		f.write("\n- Variance of accuracy: ")
		f.write(str(dict_to_print["var_accuracy"]))

		f.write("\n- Average Macro average: ")
		f.write(str(dict_to_print["macro"]))
		f.write("\n- Variance of Macro average: ")
		f.write(str(dict_to_print["var_macro"]))

		f.write("\n- Average Weighted average: ")
		f.write(str(dict_to_print["weighted"]))
		f.write("\n- Variance of Weighted average: ")
		f.write(str(dict_to_print["var_weighted"]))
		f.write("\n" + "-" * 60 + "\n")