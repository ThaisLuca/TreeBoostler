'''
   Functions to handle revision theory
   Name:         revision.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

import shutil
import os

def delete_train_files():
    try:
        shutil.rmtree('boostsrl/train')
    except:
        pass
    try:
        os.remove('boostsrl/train_output.txt')
    except:
        pass

def delete_test_files():
    try:
        shutil.rmtree('boostsrl/test')
    except:
        pass
    try:
        os.remove('boostsrl/test_output.txt')
    except:
        pass
    
def delete_model_files():
    delete_train_files()
    delete_test_files()
    
def save_model_files():
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
    shutil.move('boostsrl/best/train', 'boostsrl')
    shutil.move('boostsrl/best/test', 'boostsrl')
    shutil.move('boostsrl/best/train_output.txt', 'boostsrl')
    shutil.move('boostsrl/best/test_output.txt', 'boostsrl')
    try:
        shutil.rmtree('boostsrl/best')
    except:
        pass
    
def print_will_produced_tree(will):
    for w in will:
        print(w)
        
def get_clause(struct, path):
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
    
def bad_leaf(value, threshold):
    '''if leaf has 0 pos and 0 neg examples keeps it. What should be done?'''
    if sum(value[1:]) == 0:
        return False
    return max(value[1:])/sum(value[1:]) < threshold

def get_bad_leaves(struct, threshold):
    leaves = struct[2]
    bad_leaves = set()
    for path, value in leaves.items():
        falseBranch = get_branch_last_level(path, 'false')
        trueBranch = get_branch_last_level(path, 'true')
        # there are two leaves in same node
        if falseBranch in leaves and trueBranch in leaves:
            # one or both can be bad
            if bad_leaf(leaves[falseBranch], threshold) and bad_leaf(leaves[trueBranch], threshold):
                bad_leaves.add(';'.join([trueBranch,falseBranch]))
                continue
        # there is only one leaf in the same node or only one of them is bad
        # if it is a bad leaf, add it to bad_leaves
        if bad_leaf(value, threshold):
            bad_leaves.add(path)
    return list(bad_leaves)

def get_cantidates(struct, threshold):
    target = struct[0]
    nodes = struct[1]
    leaves = struct[2]
    candidates = []
    revision_points = []
    bad_leaves = get_bad_leaves(struct, threshold) 
    for bad_leaf in bad_leaves:
        new_nodes = nodes.copy()
        bad_leaf = bad_leaf.split(';')
        # two leaves in a node are bad ones
        if len(bad_leaf) == 2:
            # remove its node
            b = bad_leaf[0].split(',')
            b = ','.join(b[:-1])
            new_nodes.pop(b, None)
            candidates.append(get_refine_file([target, new_nodes, leaves]))
            revision_points.append('Remove node of clause ' + get_clause(struct, bad_leaf[0]) + ' from ' + str(bad_leaf))
        else:
            candidates.append(get_refine_file([target, nodes, leaves], bad_leaf[0]))
            revision_points.append('Allow learning subtree in ' + get_clause(struct, bad_leaf[0]) + ' from ' + str(bad_leaf[0]))
    return [candidates, revision_points]

def get_branch_with(branch, next_branch):
    if not branch:
        return next_branch
    b = branch.split(',')
    b.append(next_branch)
    return ','.join(b)
    
def get_branch_last_level(branch, new_branch):
    b = branch.split(',')
    b[-1] = new_branch
    return ','.join(b)
    
def get_refine_file(struct, forceLearningIn=None, treenumber=1):
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

def learn_test_model(background, boostsrl, target, train_pos, train_neg, facts, test_pos, test_neg, refine=None, show_tree=False, trees=1, verbose=True):
    delete_model_files()
    model = boostsrl.train(background, train_pos, train_neg, facts, refine=refine, trees=trees)
    learning_time = model.traintime()
    will = model.get_will_produced_tree()
    structured = model.get_structured_tree().copy()
    # if it is using boosted trees
    # do inference with the combined one
    if trees > 1:
        os.rename('boostsrl/train/models/bRDNs/Trees/'+ target +'Tree0.tree', 'boostsrl/train/models/bRDNs/Trees/'+ target +'Tree0_temp.tree')
        os.rename('boostsrl/train/models/bRDNs/Trees/CombinedTreesTreeFile'+ target +'.tree', 'boostsrl/train/models/bRDNs/Trees/'+ target +'Tree0.tree')
    results = boostsrl.test(model, test_pos, test_neg, facts, trees=1)
    inference_time = results.testtime()
    t_results = results.summarize_results()
    #if trees > 1:
    #    os.rename('boostsrl/train/models/bRDNs/Trees/'+ target +'Tree0.tree', 'boostsrl/train/models/bRDNs/Trees/CombinedTreesTreeFile'+ target +'.tree')
    #    os.rename('boostsrl/train/models/bRDNs/Trees/'+ target +'Tree0_temp.tree', 'boostsrl/train/models/bRDNs/Trees/'+ target +'Tree0.tree')
    if verbose:
        print('WILL-Produced Tree:')
        print_will_produced_tree(will)
        print('\n')
        print('Results:')
        print(t_results)
        print('\n')
        print('Total learning time: %s seconds' % learning_time)
        print('Total inference time: %s seconds' % inference_time)
        print('AUC ROC: %s' % t_results['AUC ROC'])
        if show_tree:
            print('\n')
            print('Tree:')
            model.tree(0, target, image=True)
    return [model, learning_time, inference_time, t_results, structured]

def theory_revision(background, boostsrl, target, r_train_pos, r_train_neg, facts, validation_pos, validation_neg, test_pos, test_neg, revision_threshold, structured_tree, max_revision_iterations=10, verbose=True):
    total_revision_time = 0
    best_aucroc = 0
    best_structured = None

    # parameter learning
    [model, learning_time, inference_time, t_results, structured] = learn_test_model(background, boostsrl, target, r_train_pos, r_train_neg, facts, validation_pos, validation_neg, refine=get_refine_file(structured_tree), verbose=verbose)
    total_revision_time += learning_time + inference_time

    best_aucroc = t_results['AUC ROC']
    best_structured = structured.copy()
    save_model_files()

    # refine candidates
    for i in range(max_revision_iterations):
        if verbose:
            print('Refining iteration %s' % str(i+1))
            print('******************************************')
        found_better = False
        candidates = get_cantidates(best_structured, revision_threshold)
        revision_points = candidates[1]
        candidates = candidates[0]
        for j in range(len(candidates)):
            if verbose:
                print('Refining candidate %s of %s' % (str(j+1), len(candidates)))
                print(revision_points[j])
                print('******************************************')
            [model, learning_time, inference_time, t_results, structured] = learn_test_model(background, boostsrl, target, r_train_pos, r_train_neg, facts, validation_pos, validation_neg, refine=candidates[j], verbose=verbose)
            total_revision_time += learning_time + inference_time
            if t_results['AUC ROC'] > best_aucroc:
                found_better = True
                best_aucroc = t_results['AUC ROC']
                best_structured = structured.copy()
                save_model_files()
        if verbose:
            print('Best Tree AUC ROC so far: %s' % best_aucroc)
            print('******************************************\n')
        if found_better == False:
            break

    # test best model
    if verbose:
        print('******************************************')
    delete_model_files()
    get_saved_model_files()
    delete_test_files()
    if verbose:
        print('Total revision time: %s' % total_revision_time)
        print('Best validation AUC ROC: %s' % best_aucroc)
    will = model.get_will_produced_tree()
    results = boostsrl.test(model, test_pos, test_neg, facts)
    inference_time = results.testtime()
    t_results = results.summarize_results()
    if verbose:
        print('WILL-Produced Tree:')
        print_will_produced_tree(will)
        print('\n')
        print('Results:')
        print(t_results)
        print('\n')
        print('Total inference time: %s seconds' % inference_time)
        print('AUC ROC: %s' % t_results['AUC ROC'])
    
    return [model, total_revision_time, inference_time, t_results, structured]
