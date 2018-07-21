#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:55:49 2018

@author: rodrigoazs
"""

import re

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
    
def get_will_produced_tree(target):
    '''Return the WILL-Produced Tree #1'''
    with open(target + '_learnedWILLregressionTrees.txt', 'r') as f:
        text = f.read()
    line = re.findall(r'%%%%%  WILL-Produced Tree #1 .* %%%%%[\s\S]*% Clauses:', text)
    splitline = (line[0].split('\n'))[2:]
    for i in range(len(splitline)):
        if splitline[i] == '% Clauses:':
            return splitline[:i-2]

def get_structured_tree(target):
    '''Use the get_will_produced_tree function to get the WILL-Produced Tree #1
       and returns it as objects with nodes, std devs and number of examples reached.'''
    def get_results(groups):
        #std dev, neg, pos
        ret = [float(groups[0].replace(',','.')), 0, 0]
        match = re.findall(r'\#pos=(\d*).*', groups[1])
        if match:
            ret[2] = int(match[0].replace('.',''))
        match = re.findall(r'\#neg=(\d*)', groups[1])
        if match:
            ret[1] = int(match[0].replace('.',''))
        return ret

    lines = get_will_produced_tree(target)
    current = []
    stack = []
    target = None
    nodes = {}
    leaves = {}
    
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
        match = re.match('.*then return [\d.-]*;\s*\/\/\s*std dev\s*=\s*([\d,.\-e]*),.*\/\*\s*(.*)\s*\*\/.*', line)
        if match:
            leaves[','.join(current)] = get_results(match.groups()) #float(match.group(1))
            if len(stack):
                current = stack.pop()
        match = re.match('.*else return [\d.-]*;\s*\/\/\s*std dev\s*=\s*([\d,.\-e]*),.*\/\*\s*(.*)\s*\*\/.*', line)
        if match:
            leaves[','.join(current)] = get_results(match.groups()) #float(match.group(1))
            if len(stack):
                current = stack.pop()
    return [target, nodes, leaves]

get_structured_tree('advisedby')
get_refine_file(get_structured_tree('advisedby'))