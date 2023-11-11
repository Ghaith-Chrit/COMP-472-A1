import os
import pandas as pd
from internal import data_visualizations, base_dt, top_dt, base_mlp, top_mlp, preprocessor,utils
from sklearn.model_selection import train_test_split

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

data_visualizations.create_percentage_plots(penguin_df, penguin_graphic_file_name)
data_visualizations.create_percentage_plots(abalone_df, abalone_graphic_file_name)

penguin_df = preprocessor.preprocess_data(penguin_df, penguin_target_col_name, one_hot_vector=True)
abalone_df = preprocessor.preprocess_data(abalone_df, abalone_target_col_name, one_hot_vector=True)

#splitting data for penguin 
X, y = penguin_df.drop([penguin_target_col_name], axis=1), penguin_df[penguin_target_col_name]
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, y)
class_names_p: list = y.unique().astype('str').tolist()

#splitting data for abalone
X, y = abalone_df.drop([abalone_target_col_name], axis=1), abalone_df[abalone_target_col_name]
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X, y)
class_names_a: list = y.unique().astype('str').tolist()

i=0
base_dt_array_peng =[]
base_dt_array_abalone =[]
top_dt_array_peng =[]
top_dt_array_abalone =[]

base_mlp_array_peng =[]
base_mlp_array_abalone =[]
top_mlp_array_peng =[]
top_mlp_array_abalone =[]

while i<5:

    base_dt_array_peng.append(base_dt.train_and_test(Xp_train, Xp_test, yp_train, yp_test, class_names_p, penguin_base_dt_tree_file_name, penguin_output_file_name))
    base_dt_array_abalone.append(base_dt.train_and_test(Xa_train, Xa_test, ya_train, ya_test, class_names_a, abalone_base_dt_tree_file_name, abalone_output_file_name, graph_max_depth=6))
    
    top_dt_array_peng.append(top_dt.train_and_test(Xp_train, Xp_test, yp_train, yp_test , class_names_p, penguin_top_dt_tree_file_name, penguin_output_file_name))
    top_dt_array_abalone.append(top_dt.train_and_test(Xa_train, Xa_test, ya_train, ya_test, class_names_a, abalone_top_dt_tree_file_name, abalone_output_file_name, graph_max_depth=6))

    base_mlp_array_peng.append(base_mlp.train_and_test(Xp_train, Xp_test, yp_train, yp_test , class_names_p, penguin_output_file_name))
    base_mlp_array_abalone.append(base_mlp.train_and_test(Xa_train, Xa_test, ya_train, ya_test, class_names_a, abalone_output_file_name))

    top_mlp_array_peng.append(top_mlp.train_and_test(Xp_train, Xp_test, yp_train, yp_test , class_names_p, penguin_output_file_name))
    top_mlp_array_abalone.append(top_mlp.train_and_test(Xa_train, Xa_test, ya_train, ya_test, class_names_a, abalone_output_file_name))

    i += 1


base_dt_peng_answers = utils.avg_calculations(base_dt_array_peng)
base_dt_abalone_answers=utils.avg_calculations(base_dt_array_abalone)

top_dt_peng_answers=utils.avg_calculations(top_dt_array_peng)
top_dt_abalone_answers=utils.avg_calculations(top_dt_array_abalone)

base_mlp_peng_answers = utils.avg_calculations(base_mlp_array_peng)
base_dt_abalone_answers=utils.avg_calculations(base_mlp_array_abalone)

top_mlp_peng_answers=utils.avg_calculations(top_mlp_array_peng)
top_mlp_abalone_answers=utils.avg_calculations(top_mlp_array_abalone)

utils.write_q6_ans(penguin_output_file_name,base_dt_array_peng,type_classifier="baseDT")
utils.write_q6_ans(penguin_output_file_name,top_dt_array_peng,type_classifier="topDT")
utils.write_q6_ans(penguin_output_file_name,base_mlp_array_peng,type_classifier="baseMLP")
utils.write_q6_ans(penguin_output_file_name,top_mlp_array_peng,type_classifier="topMLP")

utils.write_q6_ans(abalone_output_file_name,base_dt_array_abalone,type_classifier="baseDT")
utils.write_q6_ans(abalone_output_file_name,top_dt_array_abalone,type_classifier="topDT")
utils.write_q6_ans(abalone_output_file_name,base_mlp_array_abalone,type_classifier="baseMLP")
utils.write_q6_ans(abalone_output_file_name,top_mlp_array_abalone,type_classifier="topMLP")