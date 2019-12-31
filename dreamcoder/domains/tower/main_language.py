from dreamcoder.dreamcoder import *

from dreamcoder.domains.tower.towerPrimitives import ttower, executeTower, _empty_tower, TowerState, debug_primitives
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
            
    
    class ECTowerModel(object):
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
    
    ### Initialize tower grammar and PyCCG ontology from the grammar.
    
    g0 = Grammar.uniform({"new": new_primitives,
                          "old": debug_primitives,}[arguments.pop("primitives")],
                         continuationType=ttower)
    
    translator = ECTranslator(debug=True)
    translator.set_ontology(g0)
    
    
    # Debugging tests
    DEBUG_TRANSLATION = False 
    if DEBUG_TRANSLATION:
        # Test translation of EC expressions into PyCCG.
        translated = translator.translate_ec_expr('(lambda (1x3 (tower_loopM 5 (lambda (1x3 $0)) $0)))')
        print(translated)
        assert False
    
    DEBUG_TASK = True 
    if DEBUG_TASK:
        # Test PyCCG translated expressions on tasks.
        program = '(lambda (tower_loopM 5 (lambda (3x1 $0)) $0))'
        expr = 'f0_tower_loopM(f3_5,f1_3x1,fcont_cont)'
        gold_program = '(lambda (tower_loopM 5 (lambda (lambda (3x1 $0))) $0))'
        
        
        tasks = makeLanguageTasks()
        for task in [tasks[4]]:
            sentence = task.name.split()
            print(sentence)
            model = ECTowerModel(task, translator, debug=True)
            model.evaluate_ec_program(gold_program)
            translated = translator.translate_ec_expr(gold_program)
            print(translated)
            translated_back = translator.translate_expr(translated)
            print(translated_back)
            
        assert False
        
    
    
    # Dummy lexicon for lexical categories.
    # Going to start with very permissive
    # FunctionalCategory enumerate based on semantic arity. 
    lexicon = Lexicon.fromstring(r"""
        :- S:N
        
        _dummy_1 => N {}
        _dummy_2 => S/N {}
        _dummy_3 => S/N/N {}

        _dummy_4 => N/N {}
        _dummy_5 => S\N/N {}
    """, translator.ontology, include_semantics=False)
    pyccg_learner = WordLearner(lexicon, max_expr_depth=3)
    
    # EC enumeration
    tasks = makeLanguageTasks()
    # Wake generative: enumerate using the parser, check using towers, and 
    # update the lexicon.
    
    
    from pyccg.pyccg.chart import WeightedCCGChartParser, printCCGDerivation
    UPDATE_WITH_DISTANT = True
    if UPDATE_WITH_DISTANT:
        for task in [tasks[1]]:
            sentence = task.name.split()
            model = ECTowerModel(task, translator, debug=True)
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

    
    
            
            
            
            
    
    
    
