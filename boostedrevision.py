'''
   Functions to handle revision theory of boosted trees
   Name:         revision.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

import shutil
import os

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
    
def print_will_produced_tree(will):
    '''Remove files from train folder'''
    for w in will:
        print(w)
        
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
    
def is_bad_leaf(value, threshold):
    '''Defines if given leaf is bad or not (revision point).
    If leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
    if sum(value[1:]) == 0:
        return False
    return max(value[1:])/sum(value[1:]) < threshold

def get_bad_leaf_value(value):
    '''Defines if given leaf is bad or not (revision point).
    If leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
    if sum(value[1:]) == 0:
        return None
    return max(value[1:])/sum(value[1:])
               
def get_bad_leaves(struct, threshold):
    '''Get revision points (bad leaves)'''
    leaves = struct[2]
    bad_leaves = {}
    for path, value in leaves.items():
        falseBranch = get_branch_last_level(path, 'false')
        trueBranch = get_branch_last_level(path, 'true')
        # there are two leaves in same node
        if falseBranch in leaves and trueBranch in leaves:
            # one or both can be bad
            if is_bad_leaf(leaves[falseBranch], threshold) and is_bad_leaf(leaves[trueBranch], threshold):
                bad_leaves[(';'.join([trueBranch,falseBranch]))] = (get_bad_leaf_value(leaves[falseBranch]) + get_bad_leaf_value(leaves[trueBranch]))/2 
                if falseBranch in bad_leaves:
                    del bad_leaves[falseBranch]
                if trueBranch in bad_leaves:
                    del bad_leaves[trueBranch]
                continue
        # there is only one leaf in the same node or only one of them is bad
        # if it is a bad leaf, add it to bad_leaves
        if is_bad_leaf(value, threshold) and (';'.join([trueBranch,falseBranch])) not in bad_leaves :
            bad_leaves[path] = get_bad_leaf_value(value)
    ret = [(path, value) for path, value in bad_leaves.items()]
    ret.sort(key=lambda x: x[1])
    return ret

def get_candidate(struct, threshold, treenumber=1):
    '''Get candidate refining every revision point in a tree'''
    target = struct[0]
    nodes = struct[1]
    leaves = struct[2]
    bad_leaves = get_bad_leaves(struct, threshold)
    if len(bad_leaves) > 0:
        bad_leaf = bad_leaves[0][0] # get the worst revision point
        new_nodes = nodes.copy()
        bad_leaf = bad_leaf.split(';')
        # two leaves in a node are bad ones
        if len(bad_leaf) == 2:
            # remove its node
            b = bad_leaf[0].split(',')
            b = ','.join(b[:-1])
            new_nodes.pop(b, None)
            return get_refine_file([target, new_nodes, leaves], treenumber=treenumber)
        else:
            return get_refine_file([target, nodes, leaves], bad_leaf[0], treenumber=treenumber)
    else:
        return get_refine_file([target, nodes, leaves], treenumber=treenumber)
    
def get_boosted_candidate(structs, threshold):
    refine = []
    for i in range(len(structs)):
        refine += get_candidate(structs[i], threshold, i+1)
    return refine        

def get_branch_with(branch, next_branch):
    '''Append next_branch at branch'''
    if not branch:
        return next_branch
    b = branch.split(',')
    b.append(next_branch)
    return ','.join(b)
    
def get_branch_last_level(branch, new_branch):
    '''Returns a branch last level'''
    b = branch.split(',')
    b[-1] = new_branch
    return ','.join(b)
    
def get_refine_file(struct, forceLearningIn=None, treenumber=1):
    '''Generate the refine file from given tree structure'''
    target = struct[0]
    nodes = struct[1]
    #leaves = struct[2]
    tree = treenumber-1
    refine = []
    for path, value in nodes.items():
        node = target + ' :- ' + value + '.' if not path else value + '.'
        branchTrue = 'true' if get_branch_with(path, 'true') in nodes else 'true' if forceLearningIn == get_branch_with(path, 'true') else 'false'
        branchFalse = 'true' if get_branch_with(path, 'false') in nodes else 'true' if forceLearningIn == get_branch_with(path, 'false') else 'false'
        refine.append(';'.join([str(tree), path, node, branchTrue, branchFalse]))
    return refine

def get_boosted_refine_file(structs, forceLearningIn=None):
    refine = []
    for i in range(len(structs)):
        refine += get_refine_file(structs[i], treenumber=i+1, forceLearningIn=None if forceLearningIn == None else forceLearningIn[i])
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
