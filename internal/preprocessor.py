import pandas as pd

def preprocess_data_one_hot_vector(data: pd.DataFrame, target_col: str,) -> pd.DataFrame:
  col_names = data.select_dtypes(include=object).columns.difference([target_col])
  data = pd.get_dummies(data, columns=col_names)
  return data

def preprocess_data_manually(data: pd.DataFrame, target_col: str,) -> pd.DataFrame:
  col_names = data.select_dtypes(include=object).columns.difference([target_col])
  
  for col in col_names:
    unique_values = data[col].unique()
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    data[col] = data[col].map(value_to_index)
  
  return data

def preprocess_data(data: pd.DataFrame, target_col: str, one_hot_vector: bool = False) -> None:
  if one_hot_vector:
    return preprocess_data_one_hot_vector(data, target_col)
  else:
    return preprocess_data_manually(data, target_col)