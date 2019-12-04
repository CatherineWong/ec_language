from dreamcoder.dreamcoder import *

from dreamcoder.domains.tower.towerPrimitives import ttower, executeTower, _empty_tower, TowerState
from dreamcoder.domains.tower.makeTowerTasks import *
from dreamcoder.domains.tower.tower_common import renderPlan, towerLength, centerTower
from dreamcoder.utilities import *
from dreamcoder.program import Program

from dreamcoder.domains.tower.main import *

import os
import datetime

def tower_options(parser):
    parser.add_argument("--tasks",
                        choices=["old","new","language"],
                        default="old")
    parser.add_argument("--visualize",
                        default=None, type=str)
    parser.add_argument("--solutions",
                        default=None, type=str)
    parser.add_argument("--split",
                        default=1., type=float)
    parser.add_argument("--dream",
                        default=None, type=str)
    parser.add_argument("--primitives",
                        default="old", type=str,
                        choices=["new", "old"])
    
def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of tower-building tasks.
    """

    # The below global statement is required since primitives is modified within main().
    # TODO(lcary): use a function call to retrieve and declare primitives instead.
    global primitives
    import imageio
    
    from pyccg.pyccg.logic import TypeSystem, Ontology, Expression
    from pyccg.pyccg.lexicon import Lexicon
    from pyccg.pyccg.word_learner import WordLearner
    
    ####
    DEMO = False
    if DEMO:
        DSL = {}
        bricks = Program.parse("(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0))))))))")
        DSL["bricks"] = [ [bricks.runWithArguments([x,y + 4,_empty_tower,TowerState()])[1]
                           for y in range(6, 6 + 3*4, 3) ]
                          for x in [3,8] ]
        dimensionality = {}
        dimensionality["bricks"] = 2
        
        test_task = SupervisedTower('test', program=Program.parse('(lambda (3x1 $0))'))
        print(test_task.original)
        print(test_task.logLikelihood(Program.parse('(lambda (tower_loopM 5 (lambda (lambda (3x1 $0))) $0))')))
        
        tasks = makeLanguageTasks()
        task = tasks[0]
        print(task.original)
        assert False
    #####
    
    
    class ECTranslator(object):
        def __init__(self, debug=False):
            self.ontology = None
            self.grammar = None
            self.debug = debug
        
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
            if grammar.continuationType:
                pyccg_cont_type = types.new_constant(
                                "f{}_{}".format('cont', 'cont'),
                                "{}".format(grammar.continuationType))
                constants.append(pyccg_cont_type)
                if self.debug:
                    print("EC Continuation type: %s" % str(grammar.continuationType))
                    print("\tPyCCG constant: %s, %s" % (str(pyccg_cont_type), 
                                                      str(pyccg_cont_type.type)))
                
            # Convert to typed PyCCG constants and functions.          
            for i, (l, t, p) in enumerate(grammar.productions):
                if self.debug:
                    print("EC Production %d: %s, %s, %s" % (i, l, t, p))
                
                a = len(t.functionArguments())
                if a == 0:
                    pyccg_constant = types.new_constant(
                                    "f{}_{}".format(i, p),
                                    "{}".format(t))
                    constants.append(pyccg_constant)
                    if self.debug:
                        print("\tPyCCG constant: %s, %s" % (str(pyccg_constant), 
                                                          str(pyccg_constant.type)))
                else:
                    pyccg_fn = types.new_function(
                        "f{}_{}".format(i, p),
                        t.to_pyccg_array(True),
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
            if self.grammar.continuationType is not None:
                CONT_FN = 'fcont_cont'
                if ec_sexpr.find(CONT_FN) == -1:
                    return None 
                else:
                    ec_sexpr = ec_sexpr.replace(CONT_FN, '$0')
                    ec_sexpr = "(lambda %s)" % ec_sexpr
            return ec_sexpr
    
    class ECTowerModel(object):
        def __init__(self, task, translator, debug=False):
            self.debug = debug
            self.task = task
            self.translator = translator
            self.task_timeout = 10.0
            
        def evaluate(self, pyccg_lf):
            translated = translator.translate_expr(pyccg_lf)
            if translated:
                if self.debug:
                    print("%s -> %s" % (str(pyccg_lf), str(translated)))
                try:
                    p = Program.parse(translated)
                    logLikelihood = self.task.logLikelihood(p, timeout=self.task_timeout)
                    if logLikelihood >= 0:
                        if self.debug:
                            print("FOUND ", translated)
                        return True
                except Exception as e:
                    if self.debug:
                        print(e)
                    return False
                
            else:
                return False
    

    ### Initialize tower grammar and PyCCG ontology from the grammar.
    
    g0 = Grammar.uniform({"new": new_primitives,
                          "old": primitives}[arguments.pop("primitives")],
                         continuationType=ttower)
    
    translator = ECTranslator(debug=True)
    translator.set_ontology(g0)
    
    
    # Dummy lexicon for lexical categories.
    # Going to start with very permissive
    lexicon = Lexicon.fromstring(r"""
        :- S:N
        
        _dummy_noun => N {}
        _dummy_verb => S/N {}
        _dummy_adj => N/N {}
        
    """, translator.ontology, include_semantics=False)
    pyccg_learner = WordLearner(lexicon, max_expr_depth=3)
    
    # EC enumeration
    tasks = makeLanguageTasks()
    # Wake generative: enumerate using the parser, check using towers, and 
    # update the lexicon.
    
    from pyccg.pyccg.chart import WeightedCCGChartParser, printCCGDerivation
    for task in [tasks[4]:
        sentence = task.name.split()
        model = ECTowerModel(task, translator, debug=False)
        pyccg_learner.update_with_distant(sentence, model=model, answer=True)
        
        parser = pyccg_learner.make_parser()
        results = parser.parse(sentence)
        if len(results) > 0:
            printCCGDerivation(results[0])

        
        
    
    
    
    
    # Compare enumeration.
    # tasks = makeLanguageTasks()
    # for prior, _, p in g0.enumeration(Context.EMPTY, [], tasks[0].request,
    #                                  maximumDepth=5,
    #                                  upperBound=10,
    #                                  lowerBound=0):
    #     print(p)
    # 
    # for e in translator.ontology.iter_expressions(max_depth=3):
    #     translation = translator.translate_expr(e)
    #     if translation is not None:
    #         print(e)
    #         print(translation)

    
    
            
            
            
            
    
    
    
