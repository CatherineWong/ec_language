from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives, tcanvas
from math import log
from collections import OrderedDict
from program import Program

import json
import torch
import png
import time
import subprocess
import os
import torch.nn as nn

from recognition import variable

global prefix_dreams

# : Task -> feature list
class GeomFeatureCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=10):
        super(GeomFeatureCNN, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(10, 10), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.net2 = nn.Sequential(
            nn.Linear(16*5*5, 120),  # Hardocde the first one I guess :/
            nn.Linear(120, 84),
            nn.Linear(84, H)
        )

        self.mean = [0]*(256*256)
        self.count = 0
        self.sub = prefix_dreams + str(int(time.time()))

        self.outputDimensionality = H

    def forward(self, v):
        x, y = 64, 64  # Should not hardocode these.
        floatOnlyTask = list(map(float, v))
        reshaped = [floatOnlyTask[i:i+x]
                    for i in range(0, len(floatOnlyTask), y)]
        variabled = variable(reshaped).float()
        variabled = torch.unsqueeze(variabled, 0)
        variabled = torch.unsqueeze(variabled, 0)
        output = self.net1(variabled)
        output = output.view(-1, 16*5*5)
        output = self.net2(output).clamp(min=0)
        output = torch.squeeze(output)
        return output

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.examples[0][1])

    def featuresOfProgram(self, p, t):  # Won't fix for geom
        if not os.path.exists(self.sub):
                os.makedirs(self.sub)
        try:
            fname = self.sub + "/" + str(self.count) + ".png"
            evaluated = p.evaluate([])
            with open(fname + ".dream", "w") as f:
                f.write(str(p))
            with open(fname + ".LoG", "w") as f:
                f.write(evaluated)
            output = subprocess.check_output(['./geomDrawLambdaString',
                                             fname,
                                             evaluated]).decode("utf8").split("\n")
            shape = list(map(float, output[0].split(',')))
            bigShape = map(float, output[1].split(','))
        except OSError as exc:
            raise exc
        try:
            self.mean = [x+y for x, y in zip(self.mean, bigShape)]
            self.count += 1
            return self(shape)
        except ValueError:
            return None

    def renderProgram(self, p, t):  # Won't fix for geom
        if not os.path.exists(self.sub):
                os.makedirs(self.sub)
        try:
            fname = self.sub + "/" + str(self.count) + ".png"
            evaluated = p.evaluate([])
            with open(fname + ".dream", "w") as f:
                f.write(str(p))
            with open(fname + ".LoG", "w") as f:
                f.write(evaluated)
            output = subprocess.check_output(['./geomDrawLambdaString',
                                             fname + ".png",
                                             evaluated]).decode("utf8").split("\n")
            shape = list(map(float, output[0].split(',')))
            bigShape = map(float, output[1].split(','))
        except OSError as exc:
            raise exc
        try:
            self.mean = [x+y for x, y in zip(self.mean, bigShape)]
            self.count += 1
            return shape
        except ValueError:
            return None

    def finish(self):
        if self.count > 0:
            mean = [log(1+float(x/self.count)) for x in self.mean]
            mi = min(mean)
            ma = max(mean)
            mean = [(x - mi + (1/255)) / (ma - mi) for x in mean]
            img = [(int(x*254), int(x*254), int(x*254)) for x in mean]
            img = [img[i:i+256] for i in range(0, 256*256, 256)]
            img = [tuple([e for t in x for e in t]) for x in img]
            fname = self.sub+"/"+str(self.count)+"_sum.png"
            f = open(fname, 'wb')
            w = png.Writer(256, 256)
            w.write(f, img)
            f.close()
            self.mean = [0]*(256*256)
            self.count = 0


def list_options(parser):
    parser.add_argument("--target", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--reduce", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--save", type=str,
                        default=None,
                        help="Filepath output the grammar if this is a child")
    parser.add_argument("--prefix", type=str,
                        default="experimentOutputs/geom",
                        help="Filepath output the grammar if this is a child")


if __name__ == "__main__":
    args = commandlineArguments(
            steps=1000,
            a=3,
	    topK=5,
            iterations=10,
            useRecognitionModel=True,
            helmholtzRatio=0.5,
            helmholtzBatch=500,
            featureExtractor=GeomFeatureCNN,
            maximumFrontier=1000,
            CPUs=numberOfCPUs(),
            pseudoCounts=10.0,
            activation="tanh",
            extras=list_options)
    target = args.pop("target")
    red = args.pop("reduce")
    save = args.pop("save")
    prefix = args.pop("prefix")
    prefix_dreams = prefix + "/dreams/" + ('_'.join(target)) + "/"
    prefix_pickles = prefix + "/pickles/" + ('_'.join(target)) + "/"
    if not os.path.exists(prefix_dreams):
        os.makedirs(prefix_dreams)
    if not os.path.exists(prefix_pickles):
        os.makedirs(prefix_pickles)
    tasks = makeTasks(target)
    eprint("Generated", len(tasks), "tasks")

    test, train = testTrainSplit(tasks, 0.5)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    if red is not []:
        for reducing in red:
            try:
                with open(reducing, 'r') as f:
                    prods = json.load(f)
                    for e in prods:
                        e = Program.parse(e)
                        if e.isInvented:
                            primitives.append(e)
            except EOFError:
                eprint("Couldn't grab frontier from " + reducing)
            except IOError:
                eprint("Couldn't grab frontier from " + reducing)
            except json.decoder.JSONDecodeError:
                eprint("Couldn't grab frontier from " + reducing)


    primitives = list(OrderedDict((x, True) for x in primitives).keys())
    baseGrammar = Grammar.uniform(primitives)

    eprint(baseGrammar)

    fe = GeomFeatureCNN(tasks)

    for x in range(0, 500):
        program = baseGrammar.sample(tcanvas, maximumDepth=6)
        features = fe.renderProgram(program, tcanvas)
    fe.finish()

    r = explorationCompression(baseGrammar, train,
                               testingTasks=test,
                               outputPrefix=prefix_pickles,
                               compressor="rust",
                               evaluationTimeout=0.01,
                               **args)
    needsExport = [str(z)
                   for _, _, z
                   in r.grammars[-1].productions
                   if z.isInvented]
    if save is not None:
        with open(save, 'w') as f:
            json.dump(needsExport, f)
