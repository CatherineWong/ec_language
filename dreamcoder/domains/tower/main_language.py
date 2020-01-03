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
    
def main(args):
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
            self.ec_to_pyccg_names = {}
        
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
                    if isinstance(t, list):
                        return [canonicalize_polymorphic(sub_t) for sub_t in t]
                    import re
                    if isinstance(t, str):
                        t = re.sub(r't\d', grammar.fixedVariableType.name, t)
                    elif t.isPolymorphic:
                        t = t.show(False)
                        t = re.sub(r't\d', grammar.fixedVariableType.name, t)
                    return t
                    
                if a == 0:
                    p_name = 'invented' if '#' in str(p) else str(p) 
                    pyccg_name = "f{}_{}".format(i, p_name)
                    self.ec_to_pyccg_names[str(p)] = pyccg_name
                    pyccg_constant = types.new_constant(
                                    pyccg_name,
                                    "{}".format(canonicalize_polymorphic(t)))
                    constants.append(pyccg_constant)
                    if self.debug:
                        print("\tPyCCG constant: %s, %s" % (str(pyccg_constant), 
                                                          str(pyccg_constant.type)))
                else:
                    p_name = 'invented' if '#' in str(p) else str(p) 
                    pyccg_name = "f{}_{}".format(i, p_name)
                    self.ec_to_pyccg_names[str(p)] = pyccg_name
                    pyccg_fn = types.new_function(
                        pyccg_name,
                        [canonicalize_polymorphic(type) for type in t.to_pyccg_array(True)],
                        defn=None,
                        weight=-l)
                    functions.append(pyccg_fn)
                    
                    if self.debug:
                        print("\tPyCCG fn: %s, %s, %f" % (str(pyccg_fn), 
                                                          str(pyccg_fn.type),
                                                          pyccg_fn.weight))
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
        
        def translate_ec_expr(self, ec_sexpr, debug=False):
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
            
            # Post-process to rename to PyCCG names.
            import re
            for ec_name in sorted(self.ec_to_pyccg_names, key=lambda p:len(p), reverse=True):
                # Match name and () or spaces
                for t1 in '^', '\(', '\)', '\s':
                    for t2 in  '$', '\(', '\)', '\s':
                        match = r'({}){}({})'.format(t1, re.escape(ec_name), t2)
                        sub = r'\g<1>{}\g<2>'.format(self.ec_to_pyccg_names[ec_name])
                        ec_sexpr, n = re.subn(match, sub, ec_sexpr)
                        if debug and n > 0: print(ec_sexpr)
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
    
    ### Initialize tower grammar and PyCCG ontology from the grammar.
    g0 = Grammar.uniform({"new": new_primitives,
                          "old": debug_primitives,}[args.pop("primitives")],
                         continuationType=ttower)
    g0.fixedVariableType = ttower
    translator = ECTranslator(debug=True)
    translator.set_ontology(g0)
    
    l0_string= r"""
        :- S:N
        _dummy_0 => S {}
        _dummy_1 => N {}
        _dummy_2 => S\N {}
        _dummy_3 => S/N/N {}

        _dummy_4 => N/N {}
        _dummy_5 => S\N/N {}
    """
    
    
    # Debugging tests
    DEBUG_TRANSLATION = False 
    if DEBUG_TRANSLATION:
        # Test translation of EC expressions into PyCCG.
        translated = translator.translate_ec_expr('(lambda (1x3 (tower_loopM 5 (lambda (1x3 $0)) $0)))')
        print(translated)
        assert False
    
    DEBUG_TASK = False 
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
        _dummy_5 => N\N/N {}
    """, translator.ontology, include_semantics=False)
    pyccg_learner = WordLearner(lexicon, max_expr_depth=3)
    
    # EC enumeration
    tasks = makeLanguageTasks()
    # Wake generative: enumerate using the parser, check using towers, and 
    # update the lexicon.
    
    
    from pyccg.pyccg.chart import WeightedCCGChartParser, printCCGDerivation
    
    UPDATE_WITH_DISTANT = False
    if UPDATE_WITH_DISTANT:
        for task in [tasks[1]]:
            sentence = task.name.split()
            model = ECModel(task, translator, debug=True)
            pyccg_learner.update_with_distant(sentence, model=model, answer=True)
            
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence)
            if len(results) > 0:
                printCCGDerivation(results[0])

    ## Test updating with supervised.
    DEBUG_UPDATE_WITH_SUPERVISED = False
    if DEBUG_UPDATE_WITH_SUPERVISED:
        l0 = Lexicon.fromstring(l0_string, translator.ontology, include_semantics=False)
        pyccg_learner = WordLearner(l0, max_expr_depth=3)

        tasks = makeLanguageTasks()
        for task in [tasks[4]]:
            sentence = task.name.split()
            model = ECModel(task, translator, debug=True)
            print(sentence)
            
            gold_program = '(lambda (tower_loopM 5 (lambda (lambda (3x1 $0))) $0))'
            expr = translator.translate_ec_expr(gold_program)
            pyccg_learner.update_with_supervision(sentence, model, expr)
            
            results = parser.parse(sentence)
            if len(results) > 0:
                TOP_N = 5
                for result in results[:TOP_N]:
                    printCCGDerivation(result)    
    # Test wake generative with PyCCG.
    def wake_generative_with_pyccg(grammar, tasks, maximumFrontier=None,
                                    enumerationTimeout=None,CPUs=None,
                                    solver=None,evaluationTimeout=None,
                                    starting_lex=l0_string,
                                    previous_word_learner=None,
                                    distant_supervision=False,
                                    direct_supervision=True,
                                    max_expr_depth=2,
                                    max_enumerative_depth=3,
                                    expand_depth=False,
                                    only_supervise_necessary=True,
                                    dummyMonomorphicType=tint,
                                    debug=False):
        
        translator = ECTranslator(debug=debug)
        # Hack: avoid polymorphism in PyCCG.
        grammar.fixedVariableType=dummyMonomorphicType 
        translator.set_ontology(grammar) 
        starting_lexicon = Lexicon.fromstring(starting_lex, translator.ontology, include_semantics=False)
        
        starting_depth = max_expr_depth if not expand_depth else 1                           
        pyccg_learner = previous_word_learner if previous_word_learner else      WordLearner(starting_lexicon, max_expr_depth=starting_depth)
        
        # Enumerate and cache using default enumerator.
        if direct_supervision and solver != "pyccg":
            def default_wake_generative(grammar, tasks, 
                                maximumFrontier=None,
                                enumerationTimeout=None,
                                CPUs=None,
                                solver=None,
                                evaluationTimeout=None):
                from dreamcoder.enumeration import multicoreEnumeration
                defaultFrontiers, times = multicoreEnumeration(grammar, tasks, 
                                                               maximumFrontier=maximumFrontier,
                                                               enumerationTimeout=enumerationTimeout,
                                                               CPUs=CPUs,
                                                               solver=solver,
                                                               evaluationTimeout=evaluationTimeout)
                eprint("EC Generative model enumeration results:")
                eprint(Frontier.describe(defaultFrontiers))
                summaryStatistics("Generative model", [t for t in times.values() if t is not None])
        
                defaultFrontiers = {f.task : f for f in defaultFrontiers if not f.empty}
                return defaultFrontiers, times
            cachedFrontiers, _ = default_wake_generative(grammar, tasks,
                                                      solver=solver,
                                                      maximumFrontier=maximumFrontier,
                                                      enumerationTimeout=enumerationTimeout,
                                                      CPUs=CPUs,
                                                      evaluationTimeout=evaluationTimeout)
            
        
        for i, task in enumerate(tasks):
            print("Task {} of {}: {}".format(i, len(tasks), task.name))
            sentence = task.name.split()
            model = ECModel(task, translator, debug=debug)
        
            # Try distant supervision first.
            if distant_supervision:
                pyccg_learner.max_expr_depth = starting_depth
                results = pyccg_learner.update_with_distant(sentence, model, True)
                if expand_depth:
                    while(len(results) == 0 and pyccg_learner.max_expr_depth < max_expr_depth):
                        pyccg_learner.max_expr_depth = pyccg_learner.max_expr_depth + 1
                        results = pyccg_learner.update_with_distant(sentence, model, True)
                        
                
            # Enumerate and direct supervise learner on gold programs.
            if direct_supervision:
                if only_supervise_necessary and len(results) > 0: continue
                else:
                    if debug: print("\n\n\n\nDirect Supervision")
                    if solver == 'pyccg':
                        task_type = task.request.to_pyccg_array(True)
                        for expr in translator.ontology.iter_expressions(
                                        max_depth=max_enumerative_depth, 
                                        type_request=task_type):
                            if(model.evaluate(expr)):
                                pyccg_learner.update_with_supervision(sentence, model, expr)
                    else:
                        """Supervise on cached results"""
                        if task in cachedFrontiers:
                            for entry in cachedFrontiers[task].normalize().topK(maximumFrontier):
                                if debug: print("Supervising on: {}".format(str(entry.program)))
                                expr = translator.translate_ec_expr(str(entry.program))
                                pyccg_learner.update_with_supervision(sentence, model, expr) 
                        
        # Extract frontiers from best parses and report results.
        token_frontiers = {}
        total_hit = 0
        for num_task, task in enumerate(tasks):
            sentence = task.name.split()
            parser = pyccg_learner.make_parser()
            results = parser.parse(sentence, return_aux=True)[:maximumFrontier]
            
            found_result = False
            if len(results) > 0:        
                for i, result in enumerate(results):
                    # Only update on results that actually still evaluate to true.
                    parse, _ , edges = result
                    solution = parse.label()[0].semantics()
                    model = ECModel(task, translator, debug=debug)
                    if not model.evaluate(solution): continue
                    else:
                        found_result = True
                        print("\nHIT task {} of {}: {} ".format(num_task, len(tasks), task.name))
                        print("\t w\ solution: {}".format(translator.translate_expr(solution)))
                        if debug: printCCGDerivation(parse)
                        
                        for edge in edges:
                            t, s, w = edge.token()._token, edge.token().semantics(), edge.token().weight()
                            translated = translator.translate_expr(s)
                            program = Program.parse(translated)
                            type = program.infer().makeDummyMonomorphic()
                            
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
                            
                            # Report
                            print("{} -> {}: {}".format(t, program, float(w)))
            if not found_result:
                print("\nMISS task {} of {}: {} ".format(num_task, len(tasks), task.name))
                
        # Build frontiers.
        frontiers = []
        for t in token_frontiers:
            for i, request in enumerate(token_frontiers[t]):
                # 'Tasks' are words + semantic types; wraps in dummy task object.
                # name = "{}_{}".format(t, i)
                # frontiers.append(Frontier(token_frontiers[t][request], 
                #                 task=Task(name=name, request=request, examples=[])))
                for j, f in enumerate(token_frontiers[t][request]):
                    name = "{}_{}_{}".format(t, i, j)
                    frontiers.append(Frontier([f], 
                                    task=Task(name=name, request=request, examples=[])))
        return frontiers, {f : 0.0 for f in frontiers}
    
    DEBUG_WAKE_GENERATIVE = True
    if DEBUG_WAKE_GENERATIVE:
        tasks = makeLanguageTasks()
        topDownFrontiers, times = wake_generative_with_pyccg(grammar=g0, tasks=tasks,
            maximumFrontier=5,
            solver=args['solver'],
            enumerationTimeout=args['enumerationTimeout'],
            evaluationTimeout=1,
            CPUs=args['CPUs'],            
            distant_supervision=True,
            direct_supervision=True,
            only_supervise_necessary=True,
            max_enumerative_depth=4,
            max_expr_depth=3,
            expand_depth=True,
            debug=False)
        eprint("Showing the top 5 programs in each frontier being sent to the compressor:")
        for f in topDownFrontiers:
            if f.empty:
                continue
            eprint(f.task)
            for e in f.topK(5):
                eprint("%.02f\t%s" % (e.logPosterior, e.program))
            eprint()             
            
            
            
    
    
    
