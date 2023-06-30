import os
from time import time
import itertools
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
import jax.numpy as jnp
from tqdm import tqdm_notebook as tqdm

def best_group(df, groupby, metric, verbose=False):
    grouped = df.groupby(groupby)
    best_group_idx = grouped[metric].mean().idxmin()
    best_group = grouped.get_group(best_group_idx)
    query = ' & '.join([f'{col} == {v}' for col, v in zip(groupby, best_group_idx)])
    if verbose:
        print(query)
    return best_group, query


def binary_set(k):
    return np.array([[int(b) for b in "{0:0{k}b}".format(i, k=k)] for i in range(2 ** k)])


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_results(directory, description, data):
    # Write Data #
    ##############
    check_dir(directory)
    run_time = time()
    pid = os.getpid()
    df_filename = os.path.join(directory, f'{pid}_{run_time}.pkl')
    description.update({'run_time': run_time})
    df = pd.DataFrame(data)
    ldf = len(df)
    for name, value in description.items():
        if not name in df:
            df[name] = [value] * ldf

    df.to_pickle(df_filename)


def quantize_binary(x):
    return np.where(np.real(x) > 0, np.ones_like(x, dtype=int), np.zeros_like(x, dtype=int))

def quantize_spin(x):
    return np.where(np.real(x) > 0, np.ones_like(x, dtype=int), -np.ones_like(x, dtype=int))


def load_df(data_dir, ctime=0, max_entries=None, query=None, dropna=False, cols=None):
    files = os.listdir(data_dir)
    dfs = []
    models = []
    for file in tqdm(files):
        root, ext = os.path.splitext(file)

        if ext == '.pkl':
            models.append(root.strip().split)
            file_path = os.path.join(data_dir, file)
            ftime = os.path.getctime(file_path)
            if ftime < ctime:
                continue
            try:
                # df = pd.read_csv(file_path)
                df = pd.read_pickle(file_path)

            except Exception as e:
                print(e, file_path)

            if cols:
                cols = [c for c in cols if c in df]
                df = df[cols]
            if dropna:
                df = df.dropna()
            if query:
                df = df.query(query)
            if max_entries:
                df = df.iloc[:max_entries]

            dfs.append(df)

    dfs = pd.concat(dfs)
    dfs = dfs.reset_index(drop=True)
    return dfs


def remove_duplicated(df, groupby, verbose=False):
    dup = df.duplicated(groupby)
    df = df[~dup]
    if verbose:
        print(dup.sum(), df.duplicated(groupby).sum())
    return df


def get_cases(params: dict):
    keys = params.keys()
    vals = list(params.values())
    z = list(itertools.product(*vals))
    cases = [ravel_dict({k: v for k, v in zip(keys, vs)}) for vs in z]
    return cases

def ravel_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        if isinstance(k, (list, tuple)):
            for new_k, new_v in zip(k, v):
                new_dict[new_k] = new_v
        else:
            new_dict[k] = v
    return new_dict
