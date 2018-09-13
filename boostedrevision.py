'''
   Functions to handle revision theory of boosted trees
   Name:         revision.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

import shutil
import os
import re
import copy

def delete_train_files():
    '''Remove files from train folder'''
    try:
        shutil.rmtree('boostsrl/train')
    except:
        pass
    try:
        os.remove('boostsrl/train_output.txt')
    except:
        pass

def delete_test_files():
    '''Remove files from test folder'''
    try:
        shutil.rmtree('boostsrl/test')
    except:
        pass
    try:
        os.remove('boostsrl/test_output.txt')
    except:
        pass
    
def delete_model_files():
    '''Remove files of last model'''
    delete_train_files()
    delete_test_files()
    
def save_model_files():
    '''Remove files of last model as best model'''
    try:
        shutil.rmtree('boostsrl/best')
    except:
        pass
    os.mkdir('boostsrl/best')
    shutil.move('boostsrl/train', 'boostsrl/best')
    shutil.move('boostsrl/test', 'boostsrl/best')
    shutil.move('boostsrl/train_output.txt', 'boostsrl/best')
    shutil.move('boostsrl/test_output.txt', 'boostsrl/best')

def get_saved_model_files():
    '''Recover model files of best model'''
    shutil.move('boostsrl/best/train', 'boostsrl')
    shutil.move('boostsrl/best/test', 'boostsrl')
    shutil.move('boostsrl/best/train_output.txt', 'boostsrl')
    shutil.move('boostsrl/best/test_output.txt', 'boostsrl')
    try:
        shutil.rmtree('boostsrl/best')
    except:
        pass

def get_tree(path, tree, nodes, leaves):
    children = [None, None]
    split = [] if path == '' else path.split(',')
    left = ','.join(split+['true'])
    right = ','.join(split+['false'])
    print(left)
    print(right)
    if left in nodes:
        children[0] = get_tree(left, tree, nodes, leaves)
    if right in nodes:
        children[1] = get_tree(right, tree, nodes, leaves)
    if left in leaves:
        children[0] = leaves[left]
    if right in leaves:
        children[1] = leaves[right]
    return { nodes[path]: children }
    
def print_will_produced_tree(will):
    '''Remove files from train folder'''
    for w in will:
        print(w)

def node_part_of(path, leaves):
    if len(leaves) == 0:
        return False
    if path == '' and '' in leaves:
        return True
    split = path.split(',')
    for i in range(len(split)):
        if ','.join(split[:i+1]) in leaves:
            return True
    return False
   
def get_clause(struct, path):
    '''Get definite clause of given path'''
    target = struct[0]
    nodes = struct[1]
    paths = path.split(',')
    clauses = []
    for i in range(len(paths)):
        p = ','.join(paths[:i])
        t = paths[i]
        if t == 'true':
            clauses.append(nodes[p])
    return target + ' :- ' + ', '.join(clauses) + '.'
    
def is_bad_leaf(value):
    '''Defines if given leaf is bad or not (revision point).
    If leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
    if sum(value[1:]) == 0:
        return True
    return max(value[1:])/sum(value[1:]) < 1.0

def get_bad_leaf_value(value):
    '''Defines if given leaf is bad or not (revision point).
    If leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
    if sum(value[1:]) == 0:
        return 0
    return max(value[1:])/sum(value[1:])

def get_bad_leaves(struct):
    '''Get revision points (bad leaves)'''
    leaves = struct[2]
    bad_leaves = {}
    for path, value in leaves.items():
        # if it is a bad leaf, add it to bad_leaves
        if is_bad_leaf(value):
            bad_leaves[path] = get_bad_leaf_value(value)
    ret = [(path, value) for path, value in bad_leaves.items()]
    ret.sort(key=lambda x: x[1])
    return ret

def get_candidate(struct, treenumber=1):
    '''Get candidate refining every revision point in a tree'''
    target = struct[0]
    nodes = struct[1]
    leaves = struct[2]
    bad_leaves = get_bad_leaves(struct)
    set_bad_leaves = set([i[0] for i in bad_leaves])
    if len(bad_leaves) > 0:
        bad_leaf = bad_leaves[0][0] # get the worst revision point
        #new_nodes = copy.deepcopy(nodes)
        # if leaf has no example reached, remove its node
        if bad_leaves[0][1] == 0:
            return get_refine_file([target, nodes, leaves], removeNode=[get_branch_to_last_level(bad_leaf)], treenumber=treenumber)
        # two leaves in a node are bad ones
        elif get_branch_last_level(bad_leaf, 'true') in set_bad_leaves and get_branch_last_level(bad_leaf, 'false') in set_bad_leaves:
            # remove its node
            return get_refine_file([target, nodes, leaves], removeNode=[get_branch_to_last_level(bad_leaf)], treenumber=treenumber)
        else:
            # learn subtree
            return get_refine_file([target, nodes, leaves], forceLearningIn=[bad_leaf], treenumber=treenumber)
    else:
        return get_refine_file([target, nodes, leaves], treenumber=treenumber)
    
def get_boosted_candidate(structs):
    refine = []
    for i in range(len(structs)):
        refine += get_candidate(structs[i], i+1)
    return refine        

def get_branch_with(branch, next_branch):
    '''Append next_branch at branch'''
    if not branch:
        return next_branch
    b = branch.split(',')
    b.append(next_branch)
    return ','.join(b)
    
def get_branch_last_level(branch, new_branch):
    '''Returns a branch where last level has new path'''
    b = branch.split(',')
    b[-1] = new_branch
    return ','.join(b)

def get_branch_to_last_level(branch):
    '''Returns a branch without last level'''
    b = branch.split(',')
    return ','.join(b[:-1])
    
def get_refine_file(struct, forceLearningIn=[], removeNode=[], treenumber=1):
    '''Generate the refine file from given tree structure'''
    target = struct[0]
    nodes = struct[1]
    #leaves = struct[2]
    tree = treenumber-1
    refine = []
    # if first node shold be removed, then algorithm learns from scratch
    if '' in removeNode:
        return refine
    for path, value in nodes.items():
        if not node_part_of(path, removeNode):
            node = target + ' :- ' + value + '.' if not path else value + '.'
            branchTrue = 'false' if get_branch_with(path, 'true') in removeNode else 'true' if get_branch_with(path, 'true') in nodes else 'true' if get_branch_with(path, 'true') in forceLearningIn else 'false'
            branchFalse = 'false' if get_branch_with(path, 'false') in removeNode else 'true' if get_branch_with(path, 'false') in nodes else 'true' if get_branch_with(path, 'false') in forceLearningIn else 'false'
            refine.append(';'.join([str(tree), path, node, branchTrue, branchFalse]))
    return refine

def get_boosted_refine_file(structs, forceLearningIn=[], removeNode=[]):
    refine = []
    for i in range(len(structs)):
        refine += get_refine_file(structs[i], treenumber=i+1, forceLearningIn=[] if len(forceLearningIn) != len(structs) else forceLearningIn[i], removeNode=[] if len(removeNode) != len(structs) else removeNode[i])
    return refine

def learn_model(background, boostsrl, target, train_pos, train_neg, facts, refine=None, trees=10, verbose=True):
    '''Train and test a boosted or single tree'''
    delete_model_files()
    model = boostsrl.train(background, train_pos, train_neg, facts, refine=refine, trees=trees)
    will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(trees)]
    if verbose:
        for i in will:
            print(i)
    learning_time = model.traintime()
    structured = []
    for i in range(trees):
        structured.append(model.get_structured_tree(treenumber=i+1).copy())
    return [model, learning_time, structured, will]
    
def learn_test_model(background, boostsrl, target, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=None, trees=10, verbose=True):
    '''Train and test a boosted or single tree'''
    delete_model_files()
    model = boostsrl.train(background, train_pos, train_neg, train_facts, refine=refine, trees=trees)
    will = ['WILL Produced-Tree #'+str(i+1)+'\n'+('\n'.join(model.get_will_produced_tree(treenumber=i+1))) for i in range(trees)]
    if verbose:
        for i in will:
            print(i)
    learning_time = model.traintime()
    structured = []
    for i in range(trees):
        structured.append(model.get_structured_tree(treenumber=i+1).copy())
    results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=trees)
    inference_time = results.testtime()
    t_results = results.summarize_results()
    if verbose:
        print('Results')
        print('   AUC ROC   = %s' % t_results['AUC ROC'])
        print('   AUC PR    = %s' % t_results['AUC PR'])
        print('   CLL	      = %s' % t_results['CLL'])
        print('   Precision = %s at threshold = %s' % (t_results['Precision'][0], t_results['Precision'][1]))
        print('   Recall    = %s' % t_results['Recall'])
        print('   F1        = %s' % t_results['F1'])
        print('\n')
        print('Total learning time: %s seconds' % learning_time)
        print('Total inference time: %s seconds' % inference_time)
        print('AUC ROC: %s' % t_results['AUC ROC'])
        print('\n')
    return [model, learning_time, inference_time, t_results, structured, will]

def theory_revision(background, boostsrl, target, r_train_pos, r_train_neg, train_facts, validation_pos, validation_neg, test_pos, test_neg, test_facts, revision_threshold, structured_tree, trees=10, max_revision_iterations=10, verbose=True, testAfterPL=False):
    '''Function responsible for starting the theory revision process'''
    total_revision_time = 0
    best_aucroc = 0
    best_structured = None
    pl_inference_time = 0
    pl_t_results = 0

    # parameter learning
    if verbose:
        print('******************************************')
        print('Performing Parameter Learning')
        print('******************************************')
        print('Refine')
        print(get_boosted_refine_file(structured_tree))
    [model, learning_time, inference_time, t_results, structured, will] = learn_test_model(background, boostsrl, target, r_train_pos, r_train_neg, train_facts, validation_pos, validation_neg, train_facts, refine=get_boosted_refine_file(structured_tree), trees=trees, verbose=verbose)
    # saving performed parameter learning will
    #boostsrl.write_to_file(will, 'boostsrl/last_will.txt')
    #boostsrl.write_to_file([str(structured)], 'boostsrl/last_structured.txt')
    total_revision_time += learning_time + inference_time
    
    results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=trees)
    if testAfterPL:
        pl_inference_time = results.testtime()
        pl_t_results = results.summarize_results()
        if verbose:
            print('Results in test set')
            print('   AUC ROC   = %s' % t_results['AUC ROC'])
            print('   AUC PR    = %s' % t_results['AUC PR'])
            print('   CLL	      = %s' % t_results['CLL'])
            print('   Precision = %s at threshold = %s' % (t_results['Precision'][0], t_results['Precision'][1]))
            print('   Recall    = %s' % t_results['Recall'])
            print('   F1        = %s' % t_results['F1'])
            print('\n')
            print('Total inference time: %s seconds' % inference_time)
            print('AUC ROC: %s' % t_results['AUC ROC'])

    best_aucroc = t_results['AUC ROC']
    best_structured = structured.copy()
    if verbose:
        print('Structure after Parameter Learning')
        print(best_structured)
    save_model_files()

    if verbose:
        print('******************************************')
        print('Performing Theory Revision')
        print('******************************************')
    # refine candidates
    for i in range(max_revision_iterations):
        if verbose:
            print('Refining iteration %s' % str(i+1))
            print('********************************')
        found_better = False
        candidate = get_boosted_candidate(best_structured.copy(), revision_threshold)
        if verbose:
            print('Candidate for revision')
            print(candidate)
        boostsrl.write_to_file(candidate, 'boostsrl/last_candidate.txt')
        if verbose:
            print('Refining candidate')
            print('***************************')
            print('Revision points found')
            for i in range(trees):
                print('Tree #%s: %s' % (i+1, str(get_bad_leaves(best_structured[i], revision_threshold))))
            print('\n')
        [model, learning_time, inference_time, t_results, structured, will] = learn_test_model(background, boostsrl, target, r_train_pos, r_train_neg, train_facts, validation_pos, validation_neg, train_facts, trees=trees, refine=candidate, verbose=verbose)
        total_revision_time += learning_time + inference_time
        if t_results['AUC ROC'] > best_aucroc:
            found_better = True
            best_aucroc = t_results['AUC ROC']
            best_structured = structured.copy()
            save_model_files()
        if verbose:
            print('Best model AUC ROC so far: %s' % best_aucroc)
            print('\n')
        if found_better == False:
            break

    # test best model
    if verbose:
        print('******************************************')
        print('Best model found')
        print('******************************************')
    delete_model_files()
    get_saved_model_files()
    delete_test_files()
    if verbose:
        print('Total revision time: %s' % total_revision_time)
        print('Best validation AUC ROC: %s' % best_aucroc)
        print('\n')
    results = boostsrl.test(model, test_pos, test_neg, test_facts, trees=trees)
    inference_time = results.testtime()
    t_results = results.summarize_results()
    if verbose:
        print('Results')
        print('   AUC ROC   = %s' % t_results['AUC ROC'])
        print('   AUC PR    = %s' % t_results['AUC PR'])
        print('   CLL	      = %s' % t_results['CLL'])
        print('   Precision = %s at threshold = %s' % (t_results['Precision'][0], t_results['Precision'][1]))
        print('   Recall    = %s' % t_results['Recall'])
        print('   F1        = %s' % t_results['F1'])
        print('\n')
        print('Total inference time: %s seconds' % inference_time)
        print('AUC ROC: %s' % t_results['AUC ROC'])
    
    return [model, total_revision_time, inference_time, t_results, structured, pl_inference_time, pl_t_results]

def get_graph(lines):
    '''Use the get_will_produced_tree function to get the WILL-Produced Tree #1
       and returns it as objects with nodes, std devs and number of examples reached.'''
    def get_match(match):
        return '%.3f(%s)' % (float(match[0]), match[1].strip().replace('#', ''))
       
    lines = lines.split('\n')
    current = []
    stack = []
    target = None
    nodes = {}
    leaves = {}
    ids = {}
    last_id = 1
    graph = ''
   
    for line in lines:
        if not target:
            match = re.match('\s*\%\s*FOR\s*(\w+\([\w,\s]*\)):', line)
            if match:
                target = match.group(1)
        match = re.match('.*if\s*\(\s*([\w\(\),\s]*)\s*\).*', line)
        if match:
            nodes[','.join(current)] = match.group(1).strip()
            stack.append(current+['false'])
            current.append('true')
        match = re.match('.*[then|else] return ([\d.-]*);\s*\/\/\s*std dev\s*=\s*[\d,.\-e]*,.*\/\*\s*(.*)\s*\*\/.*', line)
        if match:
            leaves[','.join(current)] = get_match(match.groups()) #float(match.group(1))
            if len(stack):
                current = stack.pop()
        else:
            match = re.match('.*[then|else] return ([\d.-]*);\s*\/\/\s*.*', line)
            if match:
                leaves[','.join(current)] = get_match(match.groups()) #float(match.group(1))
                if len(stack):
                    current = stack.pop()
                   
    for key, value in nodes.items():
        ids[key] = last_id
        graph += str(last_id) + '[label = "[' + value + ']"];\n'
        last_id += 1
    for key, value in leaves.items():
        ids[key] = last_id
        graph += str(last_id) + '[shape = box,label = "' + value + '"];\n'
        last_id += 1
    for key, value in nodes.items():
        t = key.split(',')
        t = [] if len(t) == 1 and t[0] == '' else t
        current = ids[key]
        to = ids[','.join(t + ['true'])]
        graph += str(current) + ' -> ' + str(to) + '[label="True"];\n'
        to = ids[','.join(t + ['false'])]
        graph += str(current) + ' -> ' + str(to) + '[label="False"];\n'
    return 'digraph G{\n' + graph + '}'
