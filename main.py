import os
import pandas as pd
from sklearn.model_selection import train_test_split
from internal import data_visualizations, base_dt, top_dt, base_mlp, top_mlp, preprocessor, utils

# Setup Some Variables #
penguin_dataset_path: str = os.path.join(os.path.dirname(__file__), "COMP472-A1-datasets", "penguins.csv")
abalone_dataset_path: str = os.path.join(os.path.dirname(__file__), "COMP472-A1-datasets", "abalone.csv")

penguin_graphic_file_name: str = os.path.join("output", "data_visualization", "penguin", "penguin-classes")
abalone_graphic_file_name: str = os.path.join("output", "data_visualization", "abalone", "abalone-classes")

penguin_base_dt_tree_file_name: str = os.path.join("output", "base_dt_result", "penguin", "penguin-Base-DT")
abalone_base_dt_tree_file_name: str = os.path.join("output", "base_dt_result", "abalone", "abalone-Base-DT")

penguin_top_dt_tree_file_name: str = os.path.join("output", "top_dt_result", "penguin", "penguin-Top-DT")
abalone_top_dt_tree_file_name: str = os.path.join("output", "top_dt_result", "abalone", "abalone-Top-DT")

penguin_target_col_name: str = "species"
abalone_target_col_name: str = "Type"

penguin_output_file_name: str = os.path.join("output", "penguin-performance.txt")
abalone_output_file_name: str = os.path.join("output", "abalone-performance.txt")

penguin_df: pd.DataFrame = pd.read_csv(penguin_dataset_path, dtype={"species": str, "island": str, "culmen_length_mm": float, "culmen_depth_mm": float, "flipper_length_mm": float, "body_mass_g": float, "sex": str})
abalone_df: pd.DataFrame = pd.read_csv(abalone_dataset_path, dtype={"Type": str, "LongestShell": float, "Diameter": float, "Height": float, "WholeWeight": float, "ShuckedWeight": float, "VisceraWeight": float, "ShellWeight": float, "Rings": int})

# Visualize The Data #
data_visualizations.create_percentage_plots(penguin_df, penguin_graphic_file_name)
data_visualizations.create_percentage_plots(abalone_df, abalone_graphic_file_name)

# Preproces The Data #
penguin_df = preprocessor.preprocess_data(penguin_df, penguin_target_col_name, one_hot_vector=True)
abalone_df = preprocessor.preprocess_data(abalone_df, abalone_target_col_name, one_hot_vector=True)

# Split The Penguin Data #
X, y = penguin_df.drop([penguin_target_col_name], axis=1), penguin_df[penguin_target_col_name]
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, y)
class_names_p: list = y.unique().astype('str').tolist()

# Split The Abalone Data #
X, y = abalone_df.drop([abalone_target_col_name], axis=1), abalone_df[abalone_target_col_name]
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X, y)
class_names_a: list = y.unique().astype('str').tolist()

# Run The Models #
base_dt_array_peng = utils.run_model_times(model=base_dt.train_and_test, model_description="BaseDT (Penguin)", X_train=Xp_train, X_test=Xp_test, y_train=yp_train, y_test=yp_test, class_names=class_names_p, graph_file=penguin_base_dt_tree_file_name, result_file=penguin_output_file_name)
base_dt_array_abalone = utils.run_model_times(model=base_dt.train_and_test, model_description="BaseDT (Abalone)", X_train=Xa_train, X_test=Xa_test, y_train=ya_train, y_test=ya_test, class_names=class_names_a, graph_file=abalone_base_dt_tree_file_name, result_file=abalone_output_file_name, graph_max_depth=6)

top_dt_array_peng = utils.run_model_times(model=top_dt.train_and_test, model_description="TopDT (Penguin)", X_train=Xp_train, X_test=Xp_test, y_train=yp_train, y_test=yp_test, class_names=class_names_p, graph_file=penguin_top_dt_tree_file_name, result_file=penguin_output_file_name)
top_dt_array_abalone = utils.run_model_times(model=top_dt.train_and_test, model_description="TopDT (Abalone)", X_train=Xa_train, X_test=Xa_test, y_train=ya_train, y_test=ya_test, class_names=class_names_a, graph_file=abalone_top_dt_tree_file_name, result_file=abalone_output_file_name, graph_max_depth=6)

base_mlp_array_peng = utils.run_model_times(model=base_mlp.train_and_test, model_description="BaseMLP (Penguin)", X_train=Xp_train, X_test=Xp_test, y_train=yp_train, y_test=yp_test, class_names=class_names_p, result_file=penguin_output_file_name)
base_mlp_array_abalone = utils.run_model_times(model=base_mlp.train_and_test, model_description="BaseMLP (Abalone)", X_train=Xa_train, X_test=Xa_test, y_train=ya_train, y_test=ya_test, class_names=class_names_a, result_file=abalone_output_file_name)

top_mlp_array_peng = utils.run_model_times(model=top_mlp.train_and_test, model_description="TopMLP (Penguin)", X_train=Xp_train, X_test=Xp_test, y_train=yp_train, y_test=yp_test, class_names=class_names_p, result_file=penguin_output_file_name)
top_mlp_array_abalone = utils.run_model_times(model=top_mlp.train_and_test, model_description="TopMLP (Abalone)", X_train=Xa_train, X_test=Xa_test, y_train=ya_train, y_test=ya_test, class_names=class_names_a, result_file=abalone_output_file_name)

# Calculate Average And Variance #
base_dt_peng_answers = utils.calculate_average_and_variance(base_dt_array_peng)
base_dt_abalone_answers = utils.calculate_average_and_variance(base_dt_array_abalone)

top_dt_peng_answers = utils.calculate_average_and_variance(top_dt_array_peng)
top_dt_abalone_answers = utils.calculate_average_and_variance(top_dt_array_abalone)

base_mlp_peng_answers = utils.calculate_average_and_variance(base_mlp_array_peng)
base_mlp_abalone_answers = utils.calculate_average_and_variance(base_mlp_array_abalone)

top_mlp_peng_answers = utils.calculate_average_and_variance(top_mlp_array_peng)
top_mlp_abalone_answers = utils.calculate_average_and_variance(top_mlp_array_abalone)

# Save Results To A File #
utils.write_average_and_variance_to_file(penguin_output_file_name, base_dt_peng_answers, type_classifier="BaseDT")
utils.write_average_and_variance_to_file(penguin_output_file_name, top_dt_peng_answers, type_classifier="TopDT")
utils.write_average_and_variance_to_file(penguin_output_file_name, base_mlp_peng_answers, type_classifier="BaseMLP")
utils.write_average_and_variance_to_file(penguin_output_file_name, top_mlp_peng_answers, type_classifier="TopMLP")

utils.write_average_and_variance_to_file(abalone_output_file_name, base_dt_abalone_answers, type_classifier="BaseDT")
utils.write_average_and_variance_to_file(abalone_output_file_name, top_dt_abalone_answers, type_classifier="TopDT")
utils.write_average_and_variance_to_file(abalone_output_file_name, base_mlp_abalone_answers, type_classifier="BaseMLP")
utils.write_average_and_variance_to_file(abalone_output_file_name, top_mlp_abalone_answers, type_classifier="TopMLP")