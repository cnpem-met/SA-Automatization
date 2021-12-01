def separateFEsData(deviations, frontendList):
    # separatedDeviations = {fe: None for fe in frontendList}
    separatedDeviations = []
    # filtering and saving dfs
    for fe in frontendList:
        filtered_fe = deviations.filter(like=fe, axis=0)
        # separatedDeviations[fe] = filtered_fe
        separatedDeviations.append(filtered_fe)
    return separatedDeviations

def normalizeTextInBox(namesList):
    targetNumOfChar = 27
    normalizedNames = []
    for name in namesList:
        # total characters of the name
        lenght = len(name)
        # number of empty characteres of normalized name
        diff = targetNumOfChar - lenght
        numOfSideEmptyChar = int(diff/2)
        # building text
        normName = " "*numOfSideEmptyChar + name + " "*numOfSideEmptyChar
        normalizedNames.append(normName)
    return normalizedNames

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

# FALTA ADAPTAÇÃO PARA O BOOSTER
def sortFrameDictByBeamTrajectory(frameDict, accelerator="SR"):
    # sorting in alphabetical order
    keySortedList = sorted(frameDict)
    sortedFrameDict = {}
    for key in keySortedList:
        sortedFrameDict[key] = frameDict[key]

    if(accelerator == 'SR'):
        finalFrameDict = {}
        b1FrameList = []
        # correcting the case of S0xB03
        for frameName in sortedFrameDict:
            frame = sortedFrameDict[frameName]

            if ('B03-B1' in frame.name):
                b1FrameList.append(frame)
                continue

            finalFrameDict[frame.name] = frame

            if ('B03-QUAD01-NOMINAL' in frame.name):
                for b1Frame in b1FrameList:
                    finalFrameDict[b1Frame.name] = b1Frame
                b1FrameList = []
    elif(accelerator == 'booster'):
        finalFrameDict = {}
        oddWallDict = []
        evenWallDict = []
        # correcting the case of S0xB03
        for frameName in sortedFrameDict:
            frame = sortedFrameDict[frameName]
            try:
                wall_num = int(frame.name.split('-')[0][1:])
            except ValueError:
                wall_num = 0
                
            # even walls
            if (wall_num % 2 == 0):
                if ('DIP' in frame.name):
                    evenWallDict.append(frame)
                    continue
            # odd walls
            else:
                if ('DIP' in frame.name):
                    oddWallDict.append(frame)
                    continue

            finalFrameDict[frame.name] = frame

            if ('QUAD01-NOMINAL' in frame.name and wall_num % 2 == 0):
                for frame in evenWallDict:
                    finalFrameDict[frame.name] = frame
                evenWallDict = []

            if ('QUAD01-NOMINAL' in frame.name and wall_num % 2 != 0):
                for frame in oddWallDict:
                    finalFrameDict[frame.name] = frame
                oddWallDict = []
    else:
        # incluir exceções do booster aqui
        finalFrameDict = sortedFrameDict

    return finalFrameDict

def generateReport2(frameDict, accelerator):
    computedMagnets = ''
    missingMagnets = ''

    for frameName in frameDict:
        frame = frameDict[frameName]

        if ('MEASURED' in frame.name or frame.name == 'machine-local'):
            continue
        
        if (accelerator == 'SR'):
            try:
                magnetName = frame.name.split('-')[0] + '-' + frame.name.split('-')[1] + '-' + frame.name.split('-')[2]+ '-' + frame.name.split('-')[3]+'\n'
            except IndexError:
                magnetName = frame.name.split('-')[0] + '-' + frame.name.split('-')[1] + '-' + frame.name.split('-')[2]+'\n'
        else:
            magnetName = frame.name.split('-')[0] + '-' + frame.name.split('-')[1]+'\n'
        
        measuredFrame = magnetName[:-1] + '-' + 'MEASURED'

        if (not measuredFrame in frameDict):
            missingMagnets += magnetName
        else:
            computedMagnets += magnetName

    report = f'IMÃS FALTANTES\n {missingMagnets}\n\nIMÃS JÁ COMPUTADOS\n {computedMagnets}'
    
    return report

def generateReport(frameDict, accelerator):
    computedMagnets = ''
    missingMagnets = ''

    for frameName in frameDict:
        frame = frameDict[frameName]

        if ('MEASURED' in frame.name or frame.name == 'machine-local'):
            continue
        
        nameParts = []
        [nameParts.append(frame.name.split('-')[i]) for i in range (len(frame.name.split('-')) - 1)]

        magnetName = ''
        for name in nameParts:
            magnetName +=  f'-{name}'

        magnetName = magnetName[1:]
        measuredFrame = magnetName + '-MEASURED'

        if (not measuredFrame in frameDict):
            missingMagnets += f'{magnetName}\n'
        else:
            computedMagnets += f'{magnetName}\n'

    report = f'IMÃS FALTANTES\n {missingMagnets}\n\nIMÃS JÁ COMPUTADOS\n {computedMagnets}'
    
    return report