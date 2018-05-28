#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:08:11 2018

@author: rodrigoazs
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:30:31 2018

@author: Rodrigo Azevedo
"""

import os, sys
#import random
import re
#import time

types = {
'advisedby': ('person', 'person'),
'tempadvisedby': ('person', 'person'),
'taughtby': ('course', 'person'),
'ta': ('course', 'person'),
'professor': ('person'),
'student': ('person'),
'hasposition': ('person', 'faculty'),
'publication': ('title', 'person'),
#'isa': ('person','type')
}

targets = ['advisedby']
facts = list(set(types).difference(set(targets)))

os.makedirs('uwcse/train', exist_ok=True)
os.makedirs('uwcse/test', exist_ok=True)

def get_data(types):
    positives = [{}, {}, {}, {}, {}]
    negatives = [{}, {}, {}, {}, {}]
    #consts = {}
    fold = -1
    folds = {'ai':0, 'graphics': 1, 'language': 2, 'systems': 3, 'theory': 4}
    with open('../../rembedding/test/uwcselearn.pl') as f:
        for line in f:
            n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
            m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
            if m:
                relation = m.group(1).replace(' ', '')
                if relation in types:
                    entities = m.group(2).replace(' ', '').split(',')
                    fold = folds[entities[0]]
                    entities = entities[1:]
                    entities = entities[:len(types[relation])]
                    if relation in positives[fold]:
                        #relations[relation].append([entity, relation, value])
                        positives[fold][relation].append(relation + '(' + ','.join(entities) + ').')
                    else:
                        #relations[relation] = [[entity, relation, value]]
                        positives[fold][relation] = [relation + '(' + ','.join(entities) + ').']
            if n:
                relation = n.group(1).replace(' ', '')
                if relation in types:
                    entities = n.group(2).replace(' ', '').split(',')
                    fold = folds[entities[0]]
                    entities = entities[1:]
                    entities = entities[:len(types[relation])]
                    if relation in negatives[fold]:
                        #relations[relation].append([entity, relation, value])
                        negatives[fold][relation].append(relation + '(' + ','.join(entities) + ').')
                    else:
                        #relations[relation] = [[entity, relation, value]]
                        negatives[fold][relation] = [relation + '(' + ','.join(entities) + ').']
    
    return (positives, negatives)

positives, negatives = get_data(types)

train_pos = '' #open('uwcse/train/train_pos.txt', 'w')
train_neg = '' #open('uwcse/train/train_neg.txt', 'w')
test_pos = '' #open('uwcse/test/test_pos.txt', 'w')
test_neg = '' # open('uwcse/test/test_neg.txt', 'w')

for i in range(len(positives)): #-1):
    # print positive and negative targets
    for target in targets:
        text = '\n'.join(positives[i][target])
        train_pos += text + '\n'
        text = '\n'.join(negatives[i][target])
        train_neg += text + '\n'

i = len(positives)-1
for target in targets:
    text = '\n'.join(positives[i][target])
    test_pos += text + '\n'
    text = '\n'.join(negatives[i][target])
    test_neg += text + '\n'  

train_facts = '' #open('uwcse/train/train_facts.txt', 'w')
test_facts = '' # open('uwcse/test/test_facts.txt', 'w')

for i in range(len(positives)): #-1):
    # print positive and negative targets
    for fact in facts:
        text = '\n'.join(positives[i][fact])
        train_facts += text + '\n'
        
i = len(positives)-1
for fact in facts:
    text = '\n'.join(positives[i][fact])
    test_facts += text + '\n'

with open('uwcse/train/train_pos.txt', 'w') as file:
    file.write(train_pos)

with open('uwcse/train/train_neg.txt', 'w') as file:
    file.write(train_neg)

with open('uwcse/test/test_pos.txt', 'w') as file:
    file.write(test_pos)

with open('uwcse/test/test_neg.txt', 'w') as file:
    file.write(test_neg)

with open('uwcse/train/train_facts.txt', 'w') as file:
    file.write(train_facts)

with open('uwcse/test/test_facts.txt', 'w') as file:
    file.write(test_facts)
    
with open('uwcse/train/train_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')
         
with open('uwcse/train/train_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')
    
with open('uwcse/test/test_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')

with open('uwcse/background.txt', 'w') as file:    
    file.write('//Parameters\n')
    file.write('setParam: maxTreeDepth=8.\n')
    file.write('setParam: nodeSize=1.\n')
    file.write('setParam: numOfClauses=8.\n')
    file.write('//Modes\n')
    file.write('mode: professor(+name).\n')
    file.write('mode: student(+name).\n')
    file.write('mode: advisedby(+title,+person).\n'),
    file.write('mode: advisedby(+title,-person).\n'),
    file.write('mode: advisedby(-title,+person).\n'),
    file.write('mode: tempadvisedby(+title,+person).\n'),
    file.write('mode: tempadvisedby(+title,-person).\n'),
    file.write('mode: tempadvisedby(-title,+person).\n'),
    file.write('mode: taughtby(+course,+person).\n'),
    file.write('mode: taughtby(+course,-person).\n'),
    file.write('mode: taughtby(-course,+person).\n'),
    file.write('mode: ta(+course,+person).\n'),
    file.write('mode: ta(+course,-person).\n'),
    file.write('mode: ta(-course,+person).\n'),
    file.write('mode: hasposition(+person,+faculty).\n'),
    file.write('mode: hasposition(+person,-faculty).\n'),
    file.write('mode: hasposition(-person,+faculty).\n'),
    file.write('mode: publication(+title,+person).\n'),
    file.write('mode: publication(+title,-person).\n'),
    file.write('mode: publication(-title,+person).\n'),