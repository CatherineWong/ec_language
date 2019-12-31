
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.list.main_language import main, list_options
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
    args = commandlineArguments(
        enumerationTimeout=enumerationTimeout,
        recognitionTimeout=recognitionTimeout,
        iterations=iterations,
        activation='tanh',
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
        extras=list_options)
    main(args)