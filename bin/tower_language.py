try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.tower.main import TowerCNN
from dreamcoder.domains.tower.main_language import main, tower_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

DEBUG = True

if DEBUG == True:
    enumerationTimeout=5
    recognitionTimeout=5
    iterations=1
else:
    enumerationTimeout=5
    recognitionTimeout=5
    iterations=1

if __name__ == '__main__':
    arguments = commandlineArguments(
        featureExtractor=TowerCNN,
        CPUs=numberOfCPUs(),
        helmholtzRatio=0.5,
        enumerationTimeout=enumerationTimeout,
        recognitionTimeout=recognitionTimeout,
        iterations=iterations,
        a=5,
        structurePenalty=1.5,
        pseudoCounts=30,
        aic=1.0,
        topK=2,
        maximumFrontier=5,
        extras=tower_options)
    main(arguments)
