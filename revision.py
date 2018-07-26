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
    
def bad_leaf(value, threshold):
    return True if max(value[1:])/sum(value[1:]) < threshold else False

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
        else:
            candidates.append(get_refine_file([target, nodes, leaves], bad_leaf[0]))
    return candidates

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
    
def get_refine_file(struct, forceLearningIn=None):
    target = struct[0]
    nodes = struct[1]
    #leaves = struct[2]
    refine = []
    for path, value in nodes.items():
        node = target + ' :- ' + value + '.'
        branchTrue = 'true' if get_branch_with(path, 'true') in nodes else 'true' if forceLearningIn == get_branch_with(path, 'true') else 'false'
        branchFalse = 'true' if get_branch_with(path, 'false') in nodes else 'true' if forceLearningIn == get_branch_with(path, 'false') else 'false'
        refine.append(';'.join([path, node, branchTrue, branchFalse]))
    return refine
