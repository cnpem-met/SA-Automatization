from geom_entitites.frame import Frame
from geom_entitites.transformation import Transformation
from accelerator.storage_ring import SR

import pandas as pd
import numpy as np

def generateFrameDict(lookupTableDF):
    frameDict = {}

    # creating base frame
    frameDict['machine-local'] = Frame('machine-local', Transformation(
        'machine-local', 'machine-local', 0, 0, 0, 0, 0, 0))

    # iterate over dataframe
    for frameName, dof in lookupTableDF.iterrows():
        newFrameName = frameName + "-NOMINAL"
        transformation = Transformation(
            'machine-local', newFrameName, dof['Tx'], dof['Ty'], dof['Tz'], dof['Rx'], dof['Ry'], dof['Rz'])

        frame = Frame(newFrameName, transformation)
        frameDict[newFrameName] = frame

    return frameDict

def readExcel(fileDir, dataType, sheetName=0):

    # create header
    if (dataType == 'lookuptable'):
        header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    else:
        header = ['Magnet', 'x', 'y', 'z']

    # reading file
    df = pd.read_excel(fileDir, sheet_name=sheetName,
                        header=None, names=header)

    # checking if it has a bult-in header and droping it
    if (type(df.iloc[0, 1]) is str):
        df = df.drop([0])

    # sorting and indexing df
    df = df.sort_values(by=header[0])
    df.reset_index(drop=True, inplace=True)
    df = df.set_index(header[0])

    return df

def readCSV(fileDir, dataType):
    # create header
    if (dataType == 'lookuptable'):
        header = ['frame', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
    else:
        header = ['Magnet', 'x', 'y', 'z']

    # reading csv file with automatic detection of its delimeter
    data = pd.read_csv(fileDir, sep=None, engine='python', names=header)

    # checking if it has a bult-in header and droping it
    if (type(data.iloc[0, 1]) is str):
        data = data.drop([0])

    # sorting and indexing df
    data = data.sort_values(by=header[0])
    data.reset_index(drop=True, inplace=True)
    data = data.set_index(header[0])

    return data

def writeToExcel(fileDir, data, accelerator, report_type='deviations'):
    # adding extra column to hold quadrupole types
    if (report_type == 'deviations' and accelerator == 'SR'):
        data.insert(1, 'magnet-family', np.empty(len(data.iloc[:,0]), dtype=str), True)
        for magnet, dof in data.iterrows():
            magnetFamily = SR.checkMagnetFamily(magnet)
            data.at[magnet, 'magnet-family'] = magnetFamily

    data.to_excel(fileDir)
