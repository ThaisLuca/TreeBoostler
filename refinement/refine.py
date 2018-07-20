# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:36:54 2018

@author: 317005
"""
import re
    
def get_will_produced_tree():
    '''Return the WILL-Produced Tree #1'''
    with open('advisedby_learnedWILLregressionTrees.txt', 'r') as f:
        text = f.read()
    line = re.findall(r'%%%%%  WILL-Produced Tree #1 .* %%%%%[\s\S]*% Clauses:', text)
    splitline = (line[0].split('\n'))[2:]
    for i in range(len(splitline)):
        if splitline[i] == '% Clauses:':
            return splitline[:i-2]

# target, nodes, leaves        
def get_structured_tree():
    def get_results(groups):
        #std dev, neg, pos
        ret = [float(groups[0]), 0, 0]
        match = re.findall(r'\#pos=(\d*).*', groups[1])
        if match:
            ret[2] = int(match[0])
        match = re.findall(r'\#neg=(\d*)', groups[1])
        if match:
            ret[1] = int(match[0])
        return ret

    lines = get_will_produced_tree()
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
        match = re.match('.*then return [\d.-]*;\s*\/\/\s*std dev\s*=\s*([\d.\-e]*).*\/\*\s*([\#\w*=\d*\s*]*)\s*\*\/.*', line)
        if match:
            leaves[','.join(current)] = get_results(match.groups()) #float(match.group(1))
            if len(stack):
                current = stack.pop()
        match = re.match('.*else return [\d.-]*;\s*\/\/\s*std dev\s*=\s*([\d.\-e]*).*\/\*\s*([\#\w*=\d*\s*]*)\s*\*\/.*', line)
        if match:
            leaves[','.join(current)] = get_results(match.groups()) #float(match.group(1))
            if len(stack):
                current = stack.pop()
    return [target, nodes, leaves]