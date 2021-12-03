def separateFEsData(deviations, frontendList):
    # separatedDeviations = {fe: None for fe in frontendList}
    separatedDeviations = []
    # filtering and saving dfs
    for fe in frontendList:
        filtered_fe = deviations.filter(like=fe, axis=0)
        # separatedDeviations[fe] = filtered_fe
        separatedDeviations.append(filtered_fe)
    return separatedDeviations

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

def generateReport(frames_meas, frames_nom):
    computedMagnets = ''
    missingMagnets = ''

    for frame_meas in frames_meas:
        if (not frame_meas in frames_nom):
            missingMagnets += f'{frame_meas}\n'
        else:
            computedMagnets += f'{frame_meas}\n'

    report = f'IMÃS FALTANTES\n {missingMagnets}\n\nIMÃS JÁ COMPUTADOS\n {computedMagnets}'
    
    return report