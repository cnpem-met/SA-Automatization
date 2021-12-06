from typing import Dict
import pandas as pd

from geom_entitites.frame import Frame


def separate_frontend_data(deviations: pd.DataFrame, frontends: list):
    """ Auxiliary method for segregating deviation data of specific Frontends from
        the DataFrame that contains deviations of all the FEs. """

    separated_deviations = []
    # filtering and saving dfs
    for fe in frontends:
        filtered_fe = deviations.filter(like=fe, axis=0)
        separated_deviations.append(filtered_fe)

    return separated_deviations

def printDictData(objectDict, mode='console', fileName='dataDict'):
    output = ""
    for objectName in objectDict:
        output += objectName + '\n'
        magnetDict = objectDict[objectName]
        for magnet in magnetDict:
            pointDict = magnetDict[magnet].pointDict
            # shift = magnetDict[magnet].shift
            output += '\t'+magnet+'\n'
            # output += '\t\tshift: '+str(shift)+'\n'
            output += '\t\tpointlist:\n'
            for point in pointDict:
                pt = pointDict[point]
                output += '\t\t\t' + pt.name + ' ' + \
                    str(pt.x) + ' ' + str(pt.y) + \
                    ' ' + str(pt.z) + '\n'

    if (mode == 'console'):
        print(output)
    else:
        with open(f'../data/output/{fileName}.txt', 'w') as f:
            f.write(output)

def printFrameDict(frameDict):
    for frameName in frameDict:
        frame = frameDict[frameName]
        print(frame.name)
        print('\t(Tx, Ty, Tz, Rx, Ry, Rz): ('+str(frame.transformation.Tx)+', '+str(frame.transformation.Ty)+', ' +
                str(frame.transformation.Tz)+', '+str(frame.transformation.Rx)+', '+str(frame.transformation.Ry)+', '+str(frame.transformation.Rz)+')')

def generate_report(frames_meas: Dict[str, Frame], frames_nom: Dict[str, Frame]) -> str:
    """ Lists measured magnets that still weren't evatuated. """

    computed_magnets = ''
    missing_magnets = ''

    for frame_meas in frames_meas:
        if (not frame_meas in frames_nom):
            missing_magnets += f'{frame_meas}\n'
        else:
            computed_magnets += f'{frame_meas}\n'

    report = f'IMÃS FALTANTES\n {missing_magnets}\n\nIMÃS JÁ COMPUTADOS\n {computed_magnets}'
    
    return report