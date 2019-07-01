import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations

def decode_obj(line, pos=0, decoder=JSONDecoder()):
    no_white_space_regex = re.compile(r'[^\s]')
    while True:
        match = no_white_space_regex.search(line, pos)
        if not match:
            return
        pos = match.start()
        try:
            obj, pos = decoder.raw_decode(line, pos)
        except JSONDecodeError as err:
            print('Oops! something went wrong. Error: {}'.format(err))
        yield obj

def get_obj_with_last_n_val(line, n):
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    class_label = obj['classNum']
    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    last_n_indices = np.arange(0, 60)[-n:]
    data = data.loc[last_n_indices]
    return {'id': id, 'classType': class_label, 'values': data}

def get_test_obj_with_last_n_val(line, n):
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    last_n_indices = np.arange(0, 60)[-n:]
    data = data.loc[last_n_indices]
    return {'id': id, 'values': data}

def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]

def convert_json_data_to_csv(data_dir: str, file_name: str):
    """
    Generates a dataframe by concatenating the last values of each
    multi-variate time series. This method is designed as an example
    to show how a json object can be converted into a csv file.
    :param data_dir: the path to the data directory.
    :param file_name: name of the file to be read, with the extension.
    :return: the generated dataframe.
    """
    fname = os.path.join(data_dir, file_name)

    all_df, labels, ids = [], [], []
    number = 60
    with open(fname, 'r') as infile: # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_obj_with_last_n_val(line, number)
            all_df.append(obj['values'])
            labels.append(obj['classType'])
            ids.append(obj['id'])
            
    labels = duplicate(labels, number)
    ids = duplicate(ids, number)

    df = pd.concat(all_df).reset_index(drop=True)
    df = df.assign(LABEL=pd.Series(labels))
    df = df.assign(ID=pd.Series(ids))
    df.set_index([pd.Index(ids)])
    # Uncomment if you want to save this as CSV
    # df.to_csv(file_name + '_last_vals.csv', index=False)
    return df


def convert_test_json_data_to_csv(data_dir: str, file_name: str):

    fname = os.path.join(data_dir, file_name)

    all_df, ids = [], []
    number = 60
    with open(fname, 'r') as infile: # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_test_obj_with_last_n_val(line, number)
            all_df.append(obj['values'])
            ids.append(obj['id'])
            
    ids = duplicate(ids, number)

    df = pd.concat(all_df).reset_index(drop=True)
    df = df.assign(ID=pd.Series(ids))
    df.set_index([pd.Index(ids)])
    # Uncomment if you want to save this as CSV
    # df.to_csv(file_name + '_last_vals.csv', index=False)
    return df



