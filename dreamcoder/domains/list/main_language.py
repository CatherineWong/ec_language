import random
from collections import defaultdict
import json
import math
import os
import datetime

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS
from dreamcoder.program import Program
from dreamcoder.frontier import *


def retrieveJSONTasks(filename, features=False):
    """
    For JSON of the form:
        {"name": str,
         "type": {"input" : bool|int|list-of-bool|list-of-int,
                  "output": bool|int|list-of-bool|list-of-int},
         "examples": [{"i": data, "o": data}]}
    """
    with open(filename, "r") as f:
        loaded = json.load(f)
    TP = {
        "bool": tbool,
        "int": tint,
        "list-of-bool": tlist(tbool),
        "list-of-int": tlist(tint),
    }
    return [Task(
        item["name"],
        arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
        features=(None if not features else list_features(
            [((ex["i"],), ex["o"]) for ex in item["examples"]])),
        cache=False,
    ) for item in loaded]

def makeLanguageTasks():
    base_tasks = retrieveJSONTasks("data/list_tasks.json")
    prepend_tasks = {
        'prepend-k with k=0' : 'prepend 0',
        'prepend-k with k=1' : 'prepend 1',
        'prepend-k with k=2' : 'prepend 2',
        'prepend-k with k=3' : 'prepend 3',
        'prepend-k with k=4' : 'prepend 4',
        'prepend-k with k=5' : 'prepend 5',
    }
    
    def rename_task(task, new_names):
        task.name = new_names[task.name]
        return task
        
    tasks = [rename_task(t, prepend_tasks) for t in base_tasks if t.name in prepend_tasks]
    return tasks

def list_features(examples):
    if any(isinstance(i, int) for (i,), _ in examples):
        # obtain features for number inputs as list of numbers
        examples = [(([i],), o) for (i,), o in examples]
    elif any(not isinstance(i, list) for (i,), _ in examples):
        # can't handle non-lists
        return []
    elif any(isinstance(x, list) for (xs,), _ in examples for x in xs):
        # nested lists are hard to extract features for, so we'll
        # obtain features as if flattened
        examples = [(([x for xs in ys for x in xs],), o)
                    for (ys,), o in examples]

    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(examples[0][1])

    def mean(l): return 0 if not l else sum(l) / len(l)
    imean = [mean(i) for (i,), o in examples]
    ivar = [sum((v - imean[idx])**2
                for v in examples[idx][0][0])
            for idx in range(len(examples))]

    # DISABLED length of each input and output
    # total difference between length of input and output
    # DISABLED normalized count of numbers in input but not in output
    # total normalized count of numbers in input but not in output
    # total difference between means of input and output
    # total difference between variances of input and output
    # output type (-1=bool, 0=int, 1=list)
    # DISABLED outputs if integers, else -1s
    # DISABLED outputs if bools (-1/1), else 0s
    if ot == list:  # lists of ints or bools
        omean = [mean(o) for (i,), o in examples]
        ovar = [sum((v - omean[idx])**2
                    for v in examples[idx][1])
                for idx in range(len(examples))]

        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
        cnt_not_in_output = [cntr(i, o) for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [len(o) for (i,), o in examples]
        features.append(sum(len(i) - len(o) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(om - im for im, om in zip(imean, omean)))
        features.append(sum(ov - iv for iv, ov in zip(ivar, ovar)))
        features.append(1)
        # features += [-1 for _ in examples]
        # features += [0 for _ in examples]
    elif ot == bool:
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [-1 for _ in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += [0 for _ in examples]
        features.append(0)
        features.append(sum(imean))
        features.append(sum(ivar))
        features.append(-1)
        # features += [-1 for _ in examples]
        # features += [1 if o else -1 for o in outs]
    else:  # int
        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
        cnt_not_in_output = [cntr(i, [o]) for (i,), o in examples]
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [1 for (i,), o in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(o - im for im, o in zip(imean, outs)))
        features.append(sum(ivar))
        features.append(0)
        # features += outs
        # features += [0 for _ in examples]

    return features


def isListFunction(tp):
    try:
        Context().unify(tp, arrow(tlist(tint), t0))
        return True
    except UnificationFailure:
        return False


def isIntFunction(tp):
    try:
        Context().unify(tp, arrow(tint, t0))
        return True
    except UnificationFailure:
        return False


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    H = 64
    
    special = None

    def tokenize(self, examples):
        def sanitize(l): return [z if z in self.lexicon else "?"
                                 for z_ in l
                                 for z in (z_ if isinstance(z_, list) else [z_])]

        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            y = sanitize(y)
            if len(y) > self.maximumLength:
                return None

            serializedInputs = []
            for xi, x in enumerate(xs):
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                x = sanitize(x)
                if len(x) > self.maximumLength:
                    return None
                serializedInputs.append(x)

            tokenized.append((tuple(serializedInputs), y))

        return tokenized

    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.lexicon = set(flatten((t.examples for t in tasks + testingTasks), abort=lambda x: isinstance(
            x, str))).union({"LIST_START", "LIST_END", "?"})

        # Calculate the maximum length
        self.maximumLength = float('inf') # Believe it or not this is actually important to have here
        self.maximumLength = max(len(l)
                                 for t in tasks + testingTasks
                                 for xs, y in self.tokenize(t.examples)
                                 for l in [y] + [x for x in xs])

        self.recomputeTasks = True

        super(
            LearnedFeatureExtractor,
            self).__init__(
            lexicon=list(
                self.lexicon),
            tasks=tasks,
            cuda=cuda,
            H=self.H,
            bidirectional=True)


def train_necessary(t):
    if t.name in {"head", "is-primes", "len", "pop", "repeat-many", "tail", "keep primes", "keep squares"}:
        return True
    if any(t.name.startswith(x) for x in {
        "add-k", "append-k", "bool-identify-geq-k", "count-k", "drop-k",
        "empty", "evens", "has-k", "index-k", "is-mod-k", "kth-largest",
        "kth-smallest", "modulo-k", "mult-k", "remove-index-k",
        "remove-mod-k", "repeat-k", "replace-all-with-index-k", "rotate-k",
        "slice-k-n", "take-k",
    }):
        return "some"
    return False


def list_options(parser):
    parser.add_argument(
        "--noMap", action="store_true", default=False,
        help="Disable built-in map primitive")
    parser.add_argument(
        "--noUnfold", action="store_true", default=False,
        help="Disable built-in unfold primitive")
    parser.add_argument(
        "--noLength", action="store_true", default=False,
        help="Disable built-in length primitive")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lucas-old",
        choices=[
            "language",
            "bootstrap",
            "sorting",
            "Lucas-old",
            "Lucas-depth1",
            "Lucas-depth2",
            "Lucas-depth3"])
    parser.add_argument("--maxTasks", type=int,
                        default=None,
                        help="truncate tasks to fit within this boundary")
    parser.add_argument("--primitives",
                        default="common",
                        help="Which primitive set to use",
                        choices=["McCarthy", "base", "rich", "common", "noLength"])
    parser.add_argument("--extractor", type=str,
                        choices=["hand", "deep", "learned"],
                        default="learned")
    parser.add_argument("--split", metavar="TRAIN_RATIO",
                        type=float,
                        help="split test/train")
    parser.add_argument("-H", "--hidden", type=int,
                        default=64,
                        help="number of hidden units")
    parser.add_argument("--random-seed", type=int, default=17)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    random.seed(args.pop("random_seed"))

    dataset = args.pop("dataset")
    tasks = {
        "language": makeLanguageTasks,
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "bootstrap": make_list_bootstrap_tasks,
        "sorting": sortBootstrap,
        "Lucas-depth1": lambda: retrieveJSONTasks("data/list_tasks2.json")[:105],
        "Lucas-depth2": lambda: retrieveJSONTasks("data/list_tasks2.json")[:4928],
        "Lucas-depth3": lambda: retrieveJSONTasks("data/list_tasks2.json"),
    }[dataset]()

    maxTasks = args.pop("maxTasks")
    if maxTasks and len(tasks) > maxTasks:
        necessaryTasks = []  # maxTasks will not consider these
        if dataset.startswith("Lucas2.0") and dataset != "Lucas2.0-depth1":
            necessaryTasks = tasks[:105]

        eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
        random.shuffle(tasks)
        del tasks[maxTasks:]
        tasks = necessaryTasks + tasks

    if dataset.startswith("Lucas"):
        # extra tasks for filter
        tasks.extend([
            Task("remove empty lists",
                 arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
                 [((ls,), list(filter(lambda l: len(l) > 0, ls)))
                  for _ in range(15)
                  for ls in [[[random.random() < 0.5 for _ in range(random.randint(0, 3))]
                              for _ in range(4)]]]),
            Task("keep squares",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: int(math.sqrt(x)) ** 2 == x,
                                      xs)))
                  for _ in range(15)
                  for xs in [[random.choice([0, 1, 4, 9, 16, 25])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
            Task("keep primes",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: x in {2, 3, 5, 7, 11, 13, 17,
                                                      19, 23, 29, 31, 37}, xs)))
                  for _ in range(15)
                  for xs in [[random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
        ])
        for i in range(4):
            tasks.extend([
                Task("keep eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x == i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x != i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("keep gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: not x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]])
            ])

    def isIdentityTask(t):
        return all( len(xs) == 1 and xs[0] == y for xs, y in t.examples  )
    eprint("Removed", sum(isIdentityTask(t) for t in tasks), "tasks that were just the identity function")
    tasks = [t for t in tasks if not isIdentityTask(t) ]

    prims = {"base": basePrimitives,
             "McCarthy": McCarthyPrimitives,
             "common": bootstrapTarget_extra,
             "noLength": no_length,
             "rich": primitives}[args.pop("primitives")]()
    haveLength = not args.pop("noLength")
    haveMap = not args.pop("noMap")
    haveUnfold = not args.pop("noUnfold")
    eprint(f"Including map as a primitive? {haveMap}")
    eprint(f"Including length as a primitive? {haveLength}")
    eprint(f"Including unfold as a primitive? {haveUnfold}")
    
    prims = [p for p in prims
                   if (p.name != "map" or haveMap) and \
                   (p.name != "unfold" or haveUnfold) and \
                   (p.name != "length" or haveLength)]
    
    baseGrammar = Grammar.uniform(prims)

    extractor = {
        "learned": LearnedFeatureExtractor,
    }[args.pop("extractor")]
    extractor.H = args.pop("hidden")

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/list/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "%s/list"%outputDirectory,
        "evaluationTimeout": 0.0005,
    })
    

    eprint("Got {} list tasks".format(len(tasks)))
    split = args.pop("split")
    if split:
        train_some = defaultdict(list)
        for t in tasks:
            necessary = train_necessary(t)
            if not necessary:
                continue
            if necessary == "some":
                train_some[t.name.split()[0]].append(t)
            else:
                t.mustTrain = True
        for k in sorted(train_some):
            ts = train_some[k]
            random.shuffle(ts)
            ts.pop().mustTrain = True

        test, train = testTrainSplit(tasks, split)
        if True:
            test = [t for t in test
                    if t.name not in EASYLISTTASKS]

        eprint(
            "Alotted {} tasks for training and {} for testing".format(
                len(train), len(test)))
    else:
        train = tasks
        test = []

    
    """Language Translation"""
    from pyccg.pyccg.logic import TypeSystem, Ontology, Expression
    from pyccg.pyccg.lexicon import Lexicon
    from pyccg.pyccg.word_learner import WordLearner
    
    class ECTranslator(object):
        def __init__(self, debug=False, task_continuation_type=None):
            self.ontology = None
            self.grammar = None
            self.debug = debug
            self.task_continuation_type=task_continuation_type
        
        def set_ontology(self, grammar):
            self.grammar = grammar
            self.ontology = self.make_ontology(grammar)
            
        
        def make_ontology(self, grammar):
            """Construct a PyCCG-compatible ontology from a Dreamcoder grammar."""
            # Initialize base types.    
            types = TypeSystem(grammar.base_types)
            if self.debug:
                print("EC Grammar types: " + str(grammar.base_types))
                print("PYCCG types: " + str(types._types))
            
            # Initialize constants and functions from productions.
            constants, functions = [], []
            
            # Initialize continuation type if need be.
            if grammar.continuationType or self.task_continuation_type:
                cont_type = grammar.continuationType if grammar.continuationType else self.task_continuation_type
                pyccg_cont_type = types.new_constant(
                                "f{}_{}".format('cont', 'cont'),
                                "{}".format(cont_type))
                constants.append(pyccg_cont_type)
                if self.debug:
                    print("EC Continuation type: %s" % str(cont_type))
                    print("\tPyCCG constant: %s, %s" % (str(pyccg_cont_type), 
                                                      str(pyccg_cont_type.type)))
                
            # Convert to typed PyCCG constants and functions.          
            for i, (l, t, p) in enumerate(grammar.productions):
                if self.debug:
                    print("EC Production %d: %s, %s, %s" % (i, l, t, p))
                
                a = len(t.functionArguments())
                
                def canonicalize_polymorphic(t):
                    if isinstance(t, str):
                        t = t.replace('t0', grammar.fixedVariableType.name)
                    elif t.isPolymorphic:
                        t = t.show(False)
                        t = t.replace('t0', grammar.fixedVariableType.name)
                    return t
                    
                if a == 0:
                    pyccg_constant = types.new_constant(
                                    "f{}_{}".format(i, p),
                                    "{}".format(canonicalize_polymorphic(t)))
                    constants.append(pyccg_constant)
                    if self.debug:
                        print("\tPyCCG constant: %s, %s" % (str(pyccg_constant), 
                                                          str(pyccg_constant.type)))
                else:
                    pyccg_fn = types.new_function(
                        "f{}_{}".format(i, p),
                        [canonicalize_polymorphic(type) for type in t.to_pyccg_array(True)],
                        None)
                    functions.append(pyccg_fn)
                    
                    if self.debug:
                        print("\tPyCCG fn: %s, %s" % (str(pyccg_fn), 
                                                          str(pyccg_fn.type)))
            ontology = Ontology(types, functions, constants)
            return ontology
    
        def translate_expr(self, expr):
            ec_sexpr = self.ontology.as_ec_sexpr(expr, self.grammar)
            
            # Post process to add continuation types
            if self.grammar.continuationType is not None or self.task_continuation_type is not None:
                CONT_FN = 'fcont_cont'
                if ec_sexpr.find(CONT_FN) == -1:
                    return None 
                else:
                    ec_sexpr = ec_sexpr.replace(CONT_FN, '$0')
                    ec_sexpr = "(lambda %s)" % ec_sexpr
            return ec_sexpr
        
        def translate_ec_expr(self, ec_sexpr):
            # Remove continuation type.
            if self.grammar.continuationType is not None:
                CONT_VAR, CONT_FN = '$0', 'fcont_cont'
                if ec_sexpr.find(CONT_VAR) == -1:
                    return None 
                else:
                    # Replace last instance of the continuation var.
                    def rreplace(s, old, new):
                        return (s[::-1].replace(old[::-1], new[::-1], 1))[::-1]
                    ec_sexpr = rreplace(ec_sexpr, CONT_VAR, CONT_FN)
                    ec_sexpr = ec_sexpr.replace('(lambda ', '', 1)[:-1]
            
            pyccg_expr, bound = self.ontology.read_ec_sexpr(ec_sexpr, typecheck=False)
            return pyccg_expr
    
    class ECModel(object):
        """Model: wrapper around an EC task with evaluation functions."""
        def __init__(self, task, translator, debug=False):
            self.debug = debug
            self.task = task
            self.translator = translator
            self.task_timeout = 10.0
        
        def evaluate_ec_program(self, ec_sexpr):
            # Directly evaluates an EC program.
            if self.debug:
                print("%s" % (str(ec_sexpr)))
            try:
                p = Program.parse(ec_sexpr)
                logLikelihood = self.task.logLikelihood(p, timeout=self.task_timeout)
                print(logLikelihood)
                if logLikelihood >= 0:
                    if self.debug:
                        print("FOUND ", ec_sexpr)
                    return True
            except Exception as e:
                if self.debug:
                    print(e)
                return False
            
        def evaluate(self, pyccg_lf):
            """Translates a PyCCG LF and then evaluates it on the EC task."""
            translated = translator.translate_expr(pyccg_lf)
            if translated:
                if self.debug:
                    print("%s -> %s" % (str(pyccg_lf), str(translated)))
                try:
                    p = Program.parse(translated)
                    logLikelihood = self.task.logLikelihood(p, timeout=self.task_timeout)
                    if logLikelihood >= 0.0:
                        if self.debug:
                            print("FOUND ", translated)
                        return True
                except Exception as e:
                    if self.debug:
                        print(e)
                    return False
                
            else:
                return False
    
    ### Initialize grammar and PyCCG ontology from the grammar.    
    g0 = Grammar.uniform(prims)
    g0.fixedVariableType=tint
    translator = ECTranslator(debug=True)
    translator.set_ontology(g0)
    
    # Test default enumeration.
    tasks = makeLanguageTasks()
    DEBUG_ENUMERATION = False
    if DEBUG_ENUMERATION:
        for task in [tasks[0]]:
            task_type = task.request.to_pyccg_array(True)
            print("Task type: {}".format(task_type))
            for e in translator.ontology.iter_expressions(max_depth=3, type_request=task_type):
                print("PyCCG expr: {}".format(e))
                translation = translator.translate_expr(e)
                if translation is not None:
                    print("Translated: {}".format(translation))
    
    # Test individual translations.
    DEBUG_TRANSLATION = False
    if DEBUG_TRANSLATION:
        ec_program = "(lambda (lambda ($1 (lambda (lambda (cons $1 $0))) $0)))"
        p = Program.parse(ec_program)
        print(p)
        translation = translator.translate_ec_expr(ec_program)
        print(translation)
        
    ### Test single tasks.
    DEBUG_EC_TO_PYCCG = False
    if DEBUG_EC_TO_PYCCG:
        gold_program = "(lambda (cons 0 $0))"
        tasks = makeLanguageTasks()
        for task in [tasks[0]]:
            sentence = task.name.split()
            print(sentence)
            model = ECModel(task, translator, debug=True)
            evaluated_gold = model.evaluate_ec_program(gold_program)
            print("Evaluated gold EC: {}".format(evaluated_gold))
            translated = translator.translate_ec_expr(gold_program)
            print("Translated to PYCCG: {}".format(translated))
            evaluated_translated = model.evaluate(translated)
            print("Evaluated translated: {}".format(evaluated_translated))
            translated_back = translator.translate_expr(translated)
            print("Translated to EC: {}".format(translated_back))
            evaluated_translated = model.evaluate_ec_program(translated_back)
            print("Evaluated translated EC: {}".format(evaluated_translated))

    ### Test evaluating single derivation using PyCCG parser.
    l0 = Lexicon.fromstring(r"""
        :- S:N
        
        _dummy_1 => N {}
        _dummy_2 => S/N {}
        _dummy_3 => S/N/N {}

        _dummy_4 => N/N {}
        _dummy_5 => S\N/N {}
    """, translator.ontology, include_semantics=False)
    
    from pyccg.pyccg.chart import WeightedCCGChartParser, printCCGDerivation
    
    ## Test updating with supervised.
    DEBUG_UPDATE_WITH_SUPERVISED = False
    if DEBUG_UPDATE_WITH_SUPERVISED:
        pyccg_learner = WordLearner(l0, max_expr_depth=2)
        tasks = makeLanguageTasks()
        for task in tasks:
            sentence = task.name.split()
            model = ECModel(task, translator, debug=True)
            print(sentence)
            
            
            # Find a gold program by regular enumeration.
            task_type = task.request.to_pyccg_array(True)
            print("Task type: {}".format(task_type))
            for e in translator.ontology.iter_expressions(max_depth=3, type_request=task_type):
                if(model.evaluate(e)):
                    # Update with gold.
                    pyccg_learner.update_with_supervision(sentence, model, e)
                    
            pyccg_learner.update_with_distant(sentence, model=model, answer=True)
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence)
            if len(results) <= 0:
                print("CONTINUING ON.....")
                
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence)
            if len(results) > 0:
                TOP_N = 5
                for result in results[:TOP_N]:
                    printCCGDerivation(result)
        
    
    DEBUG_UPDATE_DISTANT = False
    if DEBUG_UPDATE_DISTANT:
        pyccg_learner = WordLearner(l0, max_expr_depth=2)
        for task in [tasks[1]]:
            sentence = task.name.split()
            model = ECModel(task, translator, debug=True)
            pyccg_learner.update_with_distant(sentence, model=model, answer=True)
            
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence)
            if len(results) > 0:
                printCCGDerivation(results[0])
    
    # Test wake generative with PyCCG.
    l0 = Lexicon.fromstring(r"""
        :- S:N
        
        _dummy_1 => N {}
        _dummy_2 => S/N {}
        _dummy_3 => S/N/N {}

        _dummy_4 => N/N {}
        _dummy_5 => S\N/N {}
    """, translator.ontology, include_semantics=False)
    def wake_generative_with_pyccg(grammar, tasks, maximumFrontier=None,
                                    enumerationTimeout=None,CPUs=None,
                                    solver=None,evaluationTimeout=None,
                                    starting_lexicon=None,
                                    previous_word_learner=None,
                                    distant_supervision=False,
                                    direct_supervision=True,
                                    enumerate_with_learner=True,
                                    max_expr_depth=2,
                                    max_enumerative_depth=3,
                                    debug=False):
        translator = ECTranslator(debug=debug)
        translator.set_ontology(grammar)                            
        pyccg_learner = previous_word_learner if previous_word_learner else      WordLearner(l0, max_expr_depth=max_expr_depth)
        
        for i, task in enumerate(tasks):
            print("Task {} of {}: {}".format(i, len(tasks), task.name))
            sentence = task.name.split()
            model = ECModel(task, translator, debug=debug)
        
            # Try distant supervision first.
            if distant_supervision:
                print("Error: distant supervision not implemented.")
                assert False
                
            # Enumerate and direct supervise learner on gold programs.
            if direct_supervision:
                if enumerate_with_learner:
                    task_type = task.request.to_pyccg_array(True)
                    for expr in translator.ontology.iter_expressions(
                                    max_depth=max_enumerative_depth, 
                                    type_request=task_type):
                        if(model.evaluate(expr)):
                            pyccg_learner.update_with_supervision(sentence, model, expr)
                        
        # Report results.
        for task in tasks:
            print("Parse for task {} of {}: {}".format(i, len(tasks), task.name))
            sentence = task.name.split()
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence)
            if len(results) > 0:
                print("HIT: best parse:")
                TOP_N = 1
                for result in results[:TOP_N]:
                    printCCGDerivation(result)
            else:
                print("MISS: no parse discovered.")

        
        # Extract frontiers from best parses.
        token_frontiers = {}
        for task in tasks:
            sentence = task.name.split()
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence, return_aux=True)[:maximumFrontier]
            
            if len(results) > 0:
                for result in results:
                    _, _ , edges = result
                    for edge in edges:
                        t, s, w = edge.token()._token, edge.token().semantics(), edge.token().weight()
                        translated = translator.translate_expr(s)
                        program = Program.parse(translated)
                        
                        
                        type = program.infer()
                        # TODO: get normalized likelihood.
                        frontier_entry = [FrontierEntry(program=program,
                                                logLikelihood=float(w),
                                                logPrior=grammar.logLikelihood(type, program))]
                        # Create new frontiers for each type.
                        if t not in token_frontiers:
                            token_frontiers[t] = {
                                type : frontier_entry
                            }
                        else:
                            if type in token_frontiers[t]:
                                token_frontiers[t][type] += frontier_entry
                            else:
                                token_frontiers[t][type] = frontier_entry
        # Build frontiers.
        frontiers = []
        for t in token_frontiers:
            for i, request in enumerate(token_frontiers[t]):
                # 'Tasks' are words + semantic types; wraps in dummy task object.
                name = "{}_{}".format(t, i)
                frontiers.append(Frontier(token_frontiers[t][request], 
                                task=Task(name=name, request=request, examples=[])))
        return frontiers, {f : 0.0 for f in frontiers}
            
            
    DEBUG_WAKE_GENERATIVE = False
    if DEBUG_WAKE_GENERATIVE:
        tasks = makeLanguageTasks()
        topDownFrontiers, times = wake_generative_with_pyccg(grammar=g0, tasks=tasks,
            maximumFrontier=3,
            distant_supervision=False,
            direct_supervision=True,
            enumerate_with_learner=True,
            max_enumerative_depth=3,
            debug=False)
        eprint("Showing the top 5 programs in each frontier being sent to the compressor:")
        for f in topDownFrontiers.values():
            if f.empty:
                continue
            eprint(f.task)
            for e in f.normalize().topK(5):
                eprint("%.02f\t%s" % (e.logPosterior, e.program))
            eprint() 
    
    DEBUG_EC = True
    if DEBUG_EC:
        def custom_wake(grammar, tasks, solver, maximumFrontier, enumerationTimeout, CPUs, evaluationTimeout):
            return wake_generative_with_pyccg(
            grammar=grammar,
            tasks=tasks,
            maximumFrontier=maximumFrontier,
            distant_supervision=False,
            direct_supervision=True,
            enumerate_with_learner=True,
            max_enumerative_depth=3,
            debug=False
            )
        train = makeLanguageTasks()
        explorationCompression(g0, train, testingTasks=[], custom_wake_generative=custom_wake, **args)
        
    RUN_DEFAULT_EC=False
    if RUN_DEFAULT_EC:
        explorationCompression(baseGrammar, train, testingTasks=test, **args)
