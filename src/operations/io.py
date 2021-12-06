from accelerator.storage_ring import SR

import pandas as pd
import numpy as np

def read_excel(file_path: str, data_type: str, sheet_name=0) -> pd.DataFrame:
    """ Read data from an excel file. """

    # create header
    if (data_type == 'lookuptable'):
        header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    else:
        header = ['Magnet', 'x', 'y', 'z']

    # reading file
    df = pd.read_excel(file_path, sheet_name=sheet_name,header=None, names=header)

    # checking if it has a bult-in header and droping it
    if (type(df.iloc[0, 1]) is str):
        df = df.drop([0])

    # sorting and indexing df
    df = df.sort_values(by=header[0])
    df.reset_index(drop=True, inplace=True)
    df = df.set_index(header[0])

    return df

def read_csv(file_path: str, data_type: str) -> pd.DataFrame:
    """ Read data from an csv file. """
    
    # create header
    if (data_type == 'lookuptable'):
        header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    else:
        header = ['Magnet', 'x', 'y', 'z']

    # reading csv file with automatic detection of its delimeter
    data = pd.read_csv(file_path, sep=None, engine='python', names=header)

    # checking if it has a bult-in header and droping it
    if (type(data.iloc[0, 1]) is str):
        data = data.drop([0])

    # sorting and indexing df
    data = data.sort_values(by=header[0])
    data.reset_index(drop=True, inplace=True)
    data = data.set_index(header[0])

    return data

def write_to_excel(file_path: str, data: pd.DataFrame, acc_name: str, report_type: str = 'deviations') -> None:
    """ Saves data from a DataFrame into an excel file. """

    # adding extra column to hold quadrupole types
    if (report_type == 'deviations' and acc_name == 'SR'):
        data.insert(1, 'magnet-family', np.empty(len(data.iloc[:,0]), dtype=str), True)
        for magnet, dof in data.iterrows():
            magnet_family = SR.check_magnet_family(magnet)
            data.at[magnet, 'magnet-family'] = magnet_family

    data.to_excel(file_path)
