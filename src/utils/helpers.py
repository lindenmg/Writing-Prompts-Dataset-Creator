# Copyright 2024 Gabriel Lindenmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML
from tqdm import tqdm


def round_to_next_multiple(number, factor):
    """
    Round ``number`` to next higher integer multiple of ``factor``

    Parameters
    ----------
    number: int
    factor: int

    Returns
    -------
    int
    """
    return factor * (number // factor) + factor * (number % factor > 0)


def set_seeds(seed=1337, deterministic_cudnn=False):
    """
    Set all kind of seeds for RNGs

    Parameters
    ----------
    seed: int
    deterministic_cudnn: bool
        CUDA CuDNN libary in deterministic mode (SLOWER!!!)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        # When running on the CuDNN backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_yaml(file_path):
    """
    Load YAML files

    Parameters
    ----------
    file_path: str

    Returns
    -------
    list or dict
    """
    file = Path(file_path)

    if not file.is_file():
        raise FileNotFoundError(f"{file_path} doesn't exist!")

    yaml = YAML(typ='safe')
    data = yaml.load_all(file)
    data = [k for k in data]
    if len(data) == 1:
        data = data[0]
    elif len(data) == 0:
        warnings.warn(Warning(f"YAML file {file_path} may be empty!"))
    return data


# ToDo: ADD unit-test(s) for parallel_pandas! And other functions below...
def parallel_pandas(func, df, column, new_column=None, num_threads=1):
    """
    Parallel pandas map/apply function
    
    Parameters
    ----------
    func: function
    df: pandas.DataFrame
    column: str
        Column which the function ``func`` shall be applied to
    new_column: str
        If `None` overwrite ``column``, Else create column ``new_column``
    num_threads: int
        Number of CPU threads to use (includes hyperthreading!)
        Hyperthreading is NOT recommended: use number of CPU cores
    
    Returns
    -------
    pandas.DataFrame
    """
    if num_threads > cpu_count() or num_threads < 1:
        raise ValueError(f"`num_threads` should be within the thread "
                         f"number of your CPU, but is {num_threads}")
    if new_column in df.columns:
        raise ValueError(f"`new_column`=={new_column} does already exist "
                         f"as column in the DataFrame `df`")
    if column not in df.columns:
        raise ValueError(f"`column`=={column} is not in the DataFrame `df`")

    with Pool(num_threads) as pool:
        result = pool.map(func, tqdm(df[column].values))

        if new_column is not None:
            df[new_column] = result
        else:
            df[column] = result


def dict_raise_on_duplicates(d_ordered_pairs):
    """
    Check Python dict for duplicate keys

    Parameters
    ----------
    d_ordered_pairs: dict

    Returns
    -------
    dict
        Duplicate--keys-less version of input ``d_ordered_pairs``
    """
    ls = os.linesep + os.linesep
    message = ls + "The configuration file has a duplicate key: {:}" + ls

    d = {}
    for k, v in d_ordered_pairs:
        if k in d:
            print(message.format(k))
            warnings.warn(Warning(message))
        else:
            d[k] = v
    return d


def stop_time(start_time):
    """
    Stops the time between start_time and the current time.

    Parameters
    ----------
    start_time : posix.times_result
        The start time gotten from os.times()
    Returns
    -------
    float, posix.times_result
        - time_elapsed: The seconds plus hundredth seconds since time of start_time
        - start_time: The new start time for the time of the next section
    """
    t_end = os.times()
    time_elapsed = t_end.elapsed - start_time.elapsed
    start_time = os.times()
    return time_elapsed, start_time


def section_text(text, upper=True, show_time=True):
    """
    Displays a heading with optional display of last sections runtime.

    Before the execution a global variable named t_start needs to be
    initialised with os.times().

    Parameters
    ----------
    text : str
        The text which will be displayed in the heading
    upper : bool
        Determines, if the heading shall be converted to upper case
        letters. True by default.
    show_time : bool
        Whether the time of last section shall be shown.
    Examples
    --------
    >>> import os
    >>> t_start = os.times()
    >>> text = "This is a new section"
    >>> section_text(text)
          Runtime of last section: 0.0s
    =====  THIS IS A NEW SECTION  =============================================
     *(The heading would have 4 more '=' at the end)*
    """
    if 't_start' not in globals().keys():
        global t_start
        t_start = os.times()
    time_elapsed, t_start = stop_time(t_start)

    if show_time:
        print(os.linesep
              + os.linesep
              + f"\tRuntime of last section: {time_elapsed:,.1f}s")
    text = text.upper() if upper else text
    right_delimiter = "=" * (79 - len(text) - 10)
    print("======  " + text + "  " + right_delimiter + os.linesep)
