import pandas as pd
import numpy as np
import time
from typing import Callable, Dict
from sklearn.model_selection import KFold
from pathlib import Path
import os
import pickle
from functools import wraps

class DataManager:
    """
    A class to manage and store data in HDF5 format. This class supports:
    1. Storing data into HDF5 with hierarchical keys (three levels).
    2. Appending new data to the existing dataset.
    3. Extracting a specified number of items from the stored data.

    Attributes:
        file_path (str): The path to the HDF5 file used for storage.
    """

    def __init__(self, file_path):
        """
        Initializes the DataManager object with a specified file path.
        
        Parameters:
            file_path (str): The path to the HDF5 file.
        """
        self.file_path = file_path

    def _check_data(self, data):
        """
        Checks whether the provided data is in the correct format. 
        The data dict must have the same length for all values of List. 
        For example, data = {"prior": [item1, item2, ...], "DDM": [item1, item2, ...]}.
        """

        # check whether each value in data dict has same length
        value_lengths = [len(v) for v in data.values()]
        assert len(set(value_lengths)) == 1, "Values in data dict must have the same length." 

    def _get_ids(self):
        
        with pd.HDFStore(self.file_path) as store:

            id_dict = {path.strip("/"):subgroups for (path, subgroups, _) in store.walk() if path and subgroups}
        
        return id_dict

    def store_data(self, data):
        """
        Stores the provided data into an HDF5 file with a hierarchical structure.

        The data is organized into three levels:
        - The first level represents categories (e.g., "prior", "DDM").
        - The second level corresponds to individual entries within each category (e.g., "id0", "id1").
        - The third level contains specific keys (e.g., "param", "sim_data") that store pandas DataFrame/Series.

        This structure allows for efficient storage and retrieval of categorized, indexed, and key-value pair data.

        Parameters:
            data (dict): A dictionary where:
                - Keys are categories (e.g., "prior", "DDM").
                - Values are dictionaries, where each key is an entry index (e.g., "id0", "id1"), 
                and each value is another dictionary containing pandas DataFrames or Series 
                under specific keys (e.g., "param", "sim_data").

        Example:
        ```python
        data = {
            "prior": [
                {"param": Series(...), "sim_data": DataFrame(...)},
                {"param": Series(...), "sim_data": DataFrame(...)}
            ],
            "DDM": [
                {"param": Series(...), "sim_data": DataFrame(...)},
                {"param": Series(...), "sim_data": DataFrame(...)}
            ]
        }
        store = DataManager("data.h5")
        store.store_data(data)
        ```
        """

        self._check_data(data)

        with pd.HDFStore(self.file_path, mode='a') as store:
            for key, items in data.items():
                for idx, item in enumerate(items):
                    # Store each sub-item (param, sim_data) in the hierarchical structure
                    for sub_key, sub_item in item.items():
                        store.put(f"{key}/id{idx}/{sub_key}", sub_item, format="table", append=True)

    def add_data(self, add_data):
        """
        Appends new data to the existing dataset in the HDF5 file. 
        The new data is added under the appropriate hierarchical keys (three levels).
        
        Parameters:
            add_data (dict): A dictionary containing the new data to append, structured similarly 
                              to the initial data format.
        """

        self._check_data(add_data)

        with pd.HDFStore(self.file_path, mode='a') as store:
            for key, items in add_data.items():
                for _, item in enumerate(items):
                    idx = len(self._get_ids()[key])
                    for sub_key, sub_item in item.items():
                        # Append new sub-items (param, sim_data)
                        store.put(f"{key}/id{idx}/{sub_key}", sub_item, format="table", append=True)

    def get_data_by_key(self, key):
        """
        Retrieves data from the HDF5 file based on the specified key.
        
        Parameters:
            key (str): The key to retrieve data from.
        
        Returns:
            dict: A dictionary containing the data from the specified key.
        """
        result = {}
        with pd.HDFStore(self.file_path, mode='r') as store:
            for k in store.keys():
                if key in k:
                    result[k] = store.get(k)
        return result

    def extract_data(self, type: str):
        """
        Extract data from the HDF5 storage based on the provided extraction type.

        This function allows you to extract data in three different formats depending on the 
        type specified. The function walks through the HDF5 keys and retrieves the relevant data.

        Parameters:
            type (str): The extraction type, which can be one of the following:
                - "type1": Extract data for a specific model type (e.g., 'prior', 'DDM') grouped by 
                their subkeys (e.g., 'param', 'sim_data') for each ID.
                - "type2": Extract data for each ID across all model types, grouping the data by 
                the ID and model type.
                - "type3": Extract data for each subkey (e.g., 'param', 'sim_data') across all models 
                and IDs, and group them by subkey and ID.

        Returns:
            dict: A dictionary containing the extracted data. The structure of the dictionary depends 
                on the extraction type:
                - For "type1", the result is organized by model type and each model type contains 
                subkeys like 'param' and 'sim_data' for each ID.
                - For "type2", the result is organized by ID, where each ID contains the data for 
                each model type (e.g., 'prior', 'DDM').
                - For "type3", the result is organized by subkey (e.g., 'param', 'sim_data'), and 
                each subkey contains the data for each ID across model types.

        Examples:
            1. **Extract all data for a specific model type (e.g., "prior") with subkeys like 'param' and 'sim_data'.**
            
            If the type is `"type1"`, the result will return data for the specific model type 
            (e.g., "prior") categorized by subkeys (`'param'`, `'sim_data'`) for each ID.

            Example:
            ```python
            result = extract_data(type="type1")
            # Output (if the HDF5 contains 'prior' and 'DDM' with respective subkeys):
            {
                "prior": {
                    "id0": {"param": Series(...), "sim_data": DataFrame(...)},
                    "id1": {"param": Series(...), "sim_data": DataFrame(...)}
                },
                "DDM": {
                    "id0": {"param": Series(...), "sim_data": DataFrame(...)},
                    "id1": {"param": Series(...), "sim_data": DataFrame(...)}
                }
            }
            ```

            2. **Extract all data for each ID across all models.**
            
            If the type is `"type2"`, the result will group data by `id` and for each `id`, it will contain 
            the data from all model types (`'prior'`, `'DDM'`, etc.) with subkeys like `'param'`, `'sim_data'`.

            Example:
            ```python
            result = extract_data(type="type2")
            # Output (assuming the IDs are 'id0', 'id1' and the HDF5 contains 'prior', 'DDM'):
            {
                "id0": {
                    "prior": {"param": Series(...), "sim_data": DataFrame(...)},
                    "DDM": {"param": Series(...), "sim_data": DataFrame(...)}
                },
                "id1": {
                    "prior": {"param": Series(...), "sim_data": DataFrame(...)},
                    "DDM": {"param": Series(...), "sim_data": DataFrame(...)}
                }
            }
            ```

            3. **Extract all data for a specific subkey (e.g., "param") across all models and IDs.**

            If the type is `"type3"`, the result will group the data by subkey (e.g., `'param'`, `'sim_data'`), 
            and each subkey will contain data for each ID across model types.

            Example:
            ```python
            result = extract_data(type="type3")
            # Output (assuming the subkey is 'param' and the models are 'prior', 'DDM'):
            {
                "param": {
                    "id0": {"prior": Series(...), "DDM": Series(...)},
                    "id1": {"prior": Series(...), "DDM": Series(...)}
                },
                "sim_data": {
                    "id0": {"prior": DataFrame(...), "DDM": DataFrame(...)},
                    "id1": {"prior": DataFrame(...), "DDM": DataFrame(...)}
                }
            }
            ```

        Notes:
            - The `type` parameter must be one of the following: `"type1"`, `"type2"`, or `"type3"`.
            - The function walks through the HDF5 store and organizes the data based on the provided `type`.
            - This method assumes the HDF5 store has a structured path with model types as top-level keys, followed 
            by IDs (e.g., `id0`, `id1`), and data stored in subkeys like `'param'`, `'sim_data'`.

        """
        result = {}
        
        with pd.HDFStore(self.file_path, mode='r') as store:
            
            for (path, _, subkeys) in store.walk():
            
                if not subkeys:
                    continue
                
                path_parts = path.strip("/").split('/')
                top_key = path_parts[0]
                id_key = path_parts[1]
                # idx = int(id_key[2:])

                if type == "type1":
                    
                    tmp_dict = {}
                    for subkey in subkeys:
                    
                        if top_key not in result:
                            result[top_key] = []
                        
                        get_key = "/".join([path, subkey])
                        tmp_dict[subkey] = store.get(get_key)
                    
                    result[top_key].append(tmp_dict)

                elif type == "type2":

                    # Ensure the 'id' is added to the result
                    if id_key not in result:
                        result[id_key] = {}
                    # If the model type doesn't exist for this id, initialize it
                    if top_key not in result[id_key]:
                        result[id_key][top_key] = {}
                   
                    for subkey in subkeys:
                        # Store the data under the correct model type and sub_key (param, sim_data)
                        get_key = "/".join([path, subkey])
                        result[id_key][top_key][subkey] = store.get(get_key)
                
                elif type == "type3":

                    for subkey in subkeys: 

                        if subkey not in result:
                            result[subkey] = {}
                        if id_key not in result[subkey]:
                            result[subkey][id_key] = {}

                        get_key = "/".join([path, subkey])
                        result[subkey][id_key][top_key] = store.get(get_key)
                    
                elif type == "type4":
                    
                    if top_key not in result:
                        result[top_key] = {}

                    for subkey in subkeys:

                        if subkey not in result[top_key]:
                            result[top_key][subkey] = {}
                        
                        get_key = "/".join([path, subkey])
                        result[top_key][subkey][id_key] = store.get(get_key)

        return result

    @property
    def all_data(self):
        """
        Extracts the specified number of items from each key in the HDF5 file.
        Returns a dictionary where the keys are the same as the HDF5 keys and the values are lists 
        of dictionaries containing pandas DataFrame or Series.

        Parameters:
            num_items (int): The number of items to retrieve from each list. Default is 10.

        Returns:
            dict: A dictionary containing the first `num_items` data from each key.
        """
        
        result = self.extract_data("type1")

        return result

    def show_keys(self):
        """
        Displays the keys in the HDF5 file.
        """
        try:
            with pd.HDFStore(self.file_path, mode='r') as store:
                all_keys = store.keys()
                print(all_keys)
        
            return all_keys
        
        except Exception:
            print(f"Error: {Exception}")
            return []

class FitStore:
    def __init__(self, save_name = 'fit_store.h5'):
        
        self.save_name = Path(save_name).with_suffix('.h5')
        self.store = pd.HDFStore(self.save_name)
        if "key_list_df" not in self.store:
            self.store.put("key_list_df", pd.DataFrame(columns=["key", "key_list"]))

    @property
    def key_list_df(self):
        return self.store['key_list_df']

    def isexist(self, key: str) -> bool:
        return self.store.key_list_df['key'].str.contains(key).any()

    def remove(self, key: str):
        if self.isexist(key):
            key_list_df = self.store.key_list_df
            key_list = key_list_df[key_list_df['key'] == key]['key_list'].values[0]
            for i in key_list:
                self.store.remove(i)

            key_list_df = key_list_df.loc[key_list_df['key'] != key]
            self.store.put('key_list_df', key_list_df)

            print(f"Key '{key}' and associated key list '{key_list}' removed.")
        else:
            raise KeyError(f"key {key} not exist")

    def __setitem__(self, key: str, value: Dict[str, pd.DataFrame]):

        key_list = []
        for i, j in value.items():
            key_str = f'{key}_{i}'
            key_list.append(key_str)
            self.store.put(key_str, j)

        key_list_df = self.store.key_list_df
        key_list_df = key_list_df._append(
            pd.Series({
                "key": key,
                "key_list": key_list
            }), ignore_index=True
        )
        self.store.put('key_list_df', key_list_df)

    def __getitem__(self, key: str) -> pd.DataFrame:
        if self.isexist(key):
            key_list_df = self.store.key_list_df
            key_list = key_list_df[key_list_df['key'] == key]['key_list'].values[0]
            return {i.replace(f"{key}_", ""): self.store[i] for i in key_list}
        else:
            raise KeyError(f"key {key} not exist")

    def close_store(self):
        self.store.close()

def cache_from_file(filename=None):
    """
    A decorator that caches the result of a function to a file using pickle.
    
    If the file exists, it loads the result from the file.
    If the file does not exist, it runs the function, saves the result to the file,
    and returns the result.
    
    Args:
        filename (str): The name of the file to store or load the result.
                       If not provided, defaults to "{function_name}.pkl".
    
    Returns:
        Callable: The decorated function.
    
    Examples:
        >>> @cache_from_file('data.pkl')
        >>> def expensive_function():
        >>>     return [i**2 for i in range(10000)]
        >>> result = expensive_function()
        
        >>> @cache_from_file()
        >>> def another_function():
        >>>     return {'key': 'value'}
        >>> result = another_function()  # Saved as "another_function.pkl"
        
        >>> def my_function():
        >>>     return sum(range(100))
        >>> cached_function = cache_from_file('my_cache.pkl')(my_function)
        >>> result = cached_function()
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided filename or generate default based on function name
            cache_file = filename or f"{func.__name__}.pkl"
            func_name = func.__name__
            
            if os.path.exists(cache_file):
                print(f"Loading result of '{func_name}' from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"Running '{func_name}' and saving result to {cache_file}...")
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                return result
        return wrapper
    
    # Handle case when decorator is used without parentheses
    if callable(filename):
        func = filename
        filename = None
        return decorator(func)
    
    return decorator

def timer(func:Callable=None, label:str=""):
    """
    Count cost time of a function and add a label. 
    
    Args:
        func (Callable): function for running
        label (str): label for print
    
    Returns:
        Callable: decorated function
    
    Example: 
    >>> @timer
    >>> def test():
    >>>     time.sleep(1)
    >>> test()

    >>> @timer(label="test")
    >>> def test2():
    >>>     time.sleep(1)
    >>> test2()

    >>> test3 = timer(test, label="test3")
    >>> test3()
    """
    # when time decroator is called without arguments, return a wrapper
    if func is None:
        
        return lambda f: timer(f, label=label)
    
    # when time decroator is called with arguments, return a wrapper
    def wrapper(*args, **kwargs):
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        days = int(elapsed_time // (24 * 3600))
        elapsed_time %= (24 * 3600)
        hours = int(elapsed_time // 3600)
        elapsed_time %= 3600
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60


        time_str = f"{days} days" if days else ""
        time_str += f"{hours} hours" if hours or days else ""
        time_str += f"{minutes} mins" if minutes or hours or days else ""
        time_str += f"{seconds:.2f} secs" if seconds or minutes or hours or days else ""

        label2 =  label if label else func.__name__
        print(f"Runing {label2}: {time_str}")
        print("\n", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        return result
    
    return wrapper

def kfold_split_data(df, groups = ["subj_idx", "congruency"], K_fold = 5):
    """
    Splits the dataset into K-folds for cross-validation, grouping by the specified fields.
    
    Parameters:
    - df: pandas.DataFrame, the dataset to be split.
    - groups: list, the fields used for grouping the data (default is ["subj_idx", "congruency"]). Data is grouped by these fields.
    - K_fold: int, the number of folds for cross-validation (default is 5).
    
    Returns:
    - K_fold_datasets: dict, containing training and testing sets for each fold. Each key is 'fold_i' (e.g., 'fold_1'), and the value is a dictionary containing 'train' and 'test' DataFrames.
    
    Example:
    ```python
    import pandas as pd
    from sklearn.model_selection import KFold
    
    # Example data
    data = {
        "subj_idx": [1, 1, 2, 2, 3, 3, 4, 4],
        "congruency": [1, 0, 1, 0, 1, 0, 1, 0],
        "feature": [5, 6, 7, 8, 9, 10, 11, 12],
        "label": [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Call the function
    K_fold_datasets = kfold_split_data(df, groups=["subj_idx", "congruency"], K_fold=2)
    
    # Example output
    print(K_fold_datasets)
    ```

    Example output:
    ```python
    {
        'fold_1': {
            'train':   subj_idx  congruency  feature  label
                        2        1         7       0
                        1        0         6       1
                        4        1        10       1
                        3        0         9       0
            'test':    subj_idx  congruency  feature  label
                        1        1         5       0
                        4        0        11       0
        },
        'fold_2': {
            'train':   subj_idx  congruency  feature  label
                        1        0         6       1
                        2        1         7       0
                        3        0         9       0
                        4        1        10       1
            'test':    subj_idx  congruency  feature  label
                        1        1         5       0
                        4        0        11       0
        }
    }
    ```
    """
    kf = KFold(n_splits=K_fold, shuffle=True, random_state=1)  
    K_fold_datasets = {f'fold_{i+1}': {'train': pd.DataFrame(), 'test': pd.DataFrame()} for i in range(K_fold)}
    
    
    for _, data_i in df.groupby(groups):
        for fold, (train_index, test_index) in enumerate(kf.split(data_i)):
            
            train_df = data_i.iloc[train_index]
            test_df = data_i.iloc[test_index]
            
            K_fold_datasets[f'fold_{fold+1}']['train'] = pd.concat([K_fold_datasets[f'fold_{fold+1}']['train'], train_df])
            K_fold_datasets[f'fold_{fold+1}']['test'] = pd.concat([K_fold_datasets[f'fold_{fold+1}']['test'], test_df])

    return K_fold_datasets

def calculate_cost(empirical_props: np.array, model_props: np.array, n_param: int, n_trial: int) -> dict:
    """
    Calculate the BIC and Chi-square cost between observed and predicted data.

    Parameters:
    - obs_data: . 
    - pred_data: . 
    - n_param: Number of parameters used in the model.
    - n_trial: Number of trials.

    Returns:
    - A dictionary containing "BIC" and "chi_square" values.
    """

    # Replace zeros with small values to avoid log(0)
    empirical_props = np.maximum(empirical_props, 0.0001)
    model_props = np.maximum(model_props, 0.0001)

    # Calculate BIC
    finalsum = np.sum(n_trial * empirical_props * np.log(model_props))
    abic = -2 * finalsum + n_param * np.log(n_trial)

    # Calculate Chi-square
    chisquare = 2 * np.sum(n_trial * empirical_props * np.log(empirical_props / model_props))
    chisquare = np.nan_to_num(chisquare, nan=np.iinfo(np.int64).max)

    return {"aBIC": abic, "chi_square": chisquare}

def calculate_rmse_cost(
    caf_empirical: np.ndarray,
    delta_empirical: np.ndarray,
    caf_ppd: np.ndarray,
    delta_ppd: np.ndarray,
    **kwargs
) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) cost from separate caf and delta arrays.
    
    The function computes the RMSE for two components: `caf` and `delta`, 
    using empirical and predicted data. These components are combined with 
    specific weights to generate a final RMSE cost.
    
    Parameters:
    ----------
    caf_empirical : np.ndarray
        Empirical caf data, 1D array of length n_err.
        Contains "comp" and "incomp" values.
        
    delta_empirical : np.ndarray
        Empirical delta data, 1D array of length n_rt.
        Contains "mean_comp" and "mean_incomp" values (already divided by 1000).
        
    caf_ppd : np.ndarray
        Predicted/theoretical caf data, same shape as caf_empirical.
        
    delta_ppd : np.ndarray
        Predicted/theoretical delta data, same shape as delta_empirical.
        
    Returns:
    -------
    float
        The weighted RMSE cost combining the caf and delta components.
        
    Raises:
    ------
    ValueError
        If input arrays have incompatible shapes or dimensions.
        
    Example:
    --------
    caf_empirical = ob_res_caf[["comp", "incomp"]].to_numpy().reshape(-1)
    delta_empirical = ob_res_delta[["mean_comp", "mean_incomp"]].to_numpy().reshape(-1) / 1000
    caf_ppd = ppd_res_caf[["comp", "incomp"]].to_numpy().reshape(-1)
    delta_ppd = ppd_res_delta[["mean_comp", "mean_incomp"]].to_numpy().reshape(-1) / 1000
    
    cost = calculate_rmse_cost_arrays(
        caf_empirical, delta_empirical, caf_ppd, delta_ppd
    )
    """
    
    if len(caf_empirical) != len(caf_ppd):
        raise ValueError(
            f"caf arrays must have same length. "
            f"Empirical: {len(caf_empirical)}, PPD: {len(caf_ppd)}"
        )
    
    if len(delta_empirical) != len(delta_ppd):
        raise ValueError(
            f"delta arrays must have same length. "
            f"Empirical: {len(delta_empirical)}, PPD: {len(delta_ppd)}"
        )
    
    n_err = len(caf_empirical)
    n_rt = len(delta_empirical)
    
    caf_empirical = np.maximum(caf_empirical, 0.0001)
    delta_empirical = np.maximum(delta_empirical, 0.0001)
    caf_ppd = np.maximum(caf_ppd, 0.0001)
    delta_ppd = np.maximum(delta_ppd, 0.0001)
    
    if n_err > 0:
        caf_diff = caf_empirical - caf_ppd
        cost_caf = np.sqrt((1 / n_err) * np.sum(caf_diff ** 2))
    else:
        cost_caf = 0.0
    
    if n_rt > 0:
        delta_diff = (delta_empirical - delta_ppd) * 1000
        cost_rt = np.sqrt((1 / n_rt) * np.sum(delta_diff ** 2))
    else:
        cost_rt = 0.0
    
    if n_err + n_rt > 0:
        weight_rt = n_rt / (n_rt + n_err)
        weight_caf = (1 - weight_rt) * 1500
    else:
        weight_rt = 0.5
        weight_caf = 0.5 * 1500
    
    return (weight_caf * cost_caf) + (weight_rt * cost_rt)

def filter_common_ids(dfs, id_col='id', task_col=None):
    """
    Filter dataframes to keep only rows with common IDs across all dataframes
    
    Parameters:
    - dfs: either a list of dataframes, or a single dataframe
    - id_col: column name for subject ID (default: 'id')
    - task_col: if single dataframe provided, column to split by for creating multiple tasks (default: None)
    
    Returns:
    - list of filtered dataframes with common IDs

    # Examples:
    ## Example 1: Multiple dataframes
    filtered_dfs = filter_common_ids([df_stroop, df_simon], id_col='id')
    ## Example 2: Single dataframe with task_col
    filtered_dfs = filter_common_ids(df_combined, id_col='id', task_col='task_name')
    ## Example 3: Access individual filtered dataframes
    if isinstance(dfs, (list, tuple)):
        df_stroop_filtered, df_simon_filtered = filtered_dfs
    else:
        # When split from single dataframe, access by index
        for i, filtered_df in enumerate(filtered_dfs):
            task_name = filtered_df.attrs.get('task', f'task_{i}')
            print(f"Task {task_name}: {filtered_df[id_col].nunique()} subjects")
    """
    if isinstance(dfs, pd.DataFrame):
        # Single dataframe provided - split by task_col
        if task_col is None:
            raise ValueError("task_col must be provided when single dataframe is given")
        
        if task_col not in dfs.columns:
            raise ValueError(f"task_col '{task_col}' not found in dataframe")
        
        # Split dataframe by task_col
        unique_tasks = dfs[task_col].unique()
        task_dfs = []
        for task in unique_tasks:
            task_df = dfs[dfs[task_col] == task].copy()
            task_df.attrs['task'] = task  # Store task info as attribute
            task_dfs.append(task_df)
        
        original_dfs = task_dfs
        print(f"Split single dataframe into {len(task_dfs)} tasks based on '{task_col}': {list(unique_tasks)}")
    else:
        # List of dataframes provided
        if not isinstance(dfs, (list, tuple)):
            raise ValueError("dfs must be either a single dataframe or a list/tuple of dataframes")
        
        original_dfs = list(dfs)
        print(f"Processing {len(original_dfs)} dataframes")
    
    # Validate that id_col exists in all dataframes
    for i, df in enumerate(original_dfs):
        if id_col not in df.columns:
            raise ValueError(f"id_col '{id_col}' not found in dataframe {i+1}")
    
    # Find common IDs across all dataframes
    all_ids = [set(df[id_col].unique()) for df in original_dfs]
    common_ids = set.intersection(*all_ids)
    
    if len(common_ids) == 0:
        print("Warning: No common IDs found across all dataframes!")
        return [df[df[id_col].isin([])] for df in original_dfs]  # Return empty dataframes
    
    # Filter all dataframes to keep only common IDs
    filtered_dfs = []
    for df in original_dfs:
        filtered_df = df[df[id_col].isin(common_ids)].copy()
        filtered_dfs.append(filtered_df)
    
    # Print summary
    print(f"Original subjects in each dataframe: {[df[id_col].nunique() for df in original_dfs]}")
    print(f"Common subjects across all dataframes: {len(common_ids)}")
    print(f"Filtered subjects in each dataframe: {[df[id_col].nunique() for df in filtered_dfs]}")
    
    return filtered_dfs
