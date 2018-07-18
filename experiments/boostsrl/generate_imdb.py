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
'workedunder': ('person', 'person'),
'genre': ('person', 'genre'),
#'female': ('person', 'person'),
'actor': ('person', 'person'),
'director': ('person', 'person'),
'movie': ('movie', 'person'),
#'isa': ('person','type')
}

targets = ['workedunder']
facts = list(set(types).difference(set(targets)))

os.makedirs('imdb/train', exist_ok=True)
os.makedirs('imdb/test', exist_ok=True)

def get_data(types):
    positives = []
    negatives = []
    #consts = {}
    fold = -1
    with open('../../rembedding/test/imdb.pl') as f:
        for line in f:
            if line[:5] == 'begin':
                fold += 1
                positives.append({})
                negatives.append({})
            n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
            m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
            if m:
                relation = m.group(1).replace(' ', '')
                entities = m.group(2).replace(' ', '').split(',')
                if relation in positives[fold]:
                    #relations[relation].append([entity, relation, value])
                    positives[fold][relation].append(relation + '(' + ','.join(entities) + ').')
                else:
                    #relations[relation] = [[entity, relation, value]]
                    positives[fold][relation] = [relation + '(' + ','.join(entities) + ').']
            if n:
                relation = n.group(1).replace(' ', '')
                entities = n.group(2).replace(' ', '').split(',')
                if relation in negatives[fold]:
                    #relations[relation].append([entity, relation, value])
                    negatives[fold][relation].append(relation + '(' + ','.join(entities) + ').')
                else:
                    #relations[relation] = [[entity, relation, value]]
                    negatives[fold][relation] = [relation + '(' + ','.join(entities) + ').']
    
    return (positives, negatives)

positives, negatives = get_data(types)

train_pos = '' #= open('imdb/train/train_pos.txt', 'w')
train_neg = '' #= open('imdb/train/train_neg.txt', 'w')
test_pos = '' #= open('imdb/test/test_pos.txt', 'w')
test_neg = '' #= open('imdb/test/test_neg.txt', 'w')

for i in range(len(positives)-1):
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

train_facts = '' #= open('imdb/train/train_facts.txt', 'w')
test_facts = '' #open('imdb/test/test_facts.txt', 'w')

for i in range(len(positives)-1):
    # print positive and negative targets
    for fact in facts:
        text = '\n'.join(positives[i][fact])
        train_facts += text + '\n'
        
i = len(positives)-1
for fact in facts:
    text = '\n'.join(positives[i][fact])
    test_facts += text + '\n'

with open('imdb/train/train_pos.txt', 'w') as file:
    file.write(train_pos)

with open('imdb/train/train_neg.txt', 'w') as file:
    file.write(train_neg)

with open('imdb/test/test_pos.txt', 'w') as file:
    file.write(test_pos)

with open('imdb/test/test_neg.txt', 'w') as file:
    file.write(test_neg)

with open('imdb/train/train_facts.txt', 'w') as file:
    file.write(train_facts)

with open('imdb/test/test_facts.txt', 'w') as file:
    file.write(test_facts)
    
with open('imdb/train/train_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')
    
with open('imdb/test/test_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')

with open('imdb/background.txt', 'w') as file:    
    file.write('//Parameters\n')
    file.write('setParam: maxTreeDepth=8.\n')
    file.write('setParam: nodeSize=3.\n')
    file.write('setParam: numOfClauses=8.\n')
    file.write('//Modes\n')
    #file.write('mode: female(+person).\n')
    file.write('mode: actor(+person).\n')
    file.write('mode: director(+person).\n')
    file.write('mode: genre(+person,+genre).\n'),
    file.write('mode: genre(+person,-genre).\n'),
    file.write('mode: genre(-person,+genre).\n'),
    file.write('mode: movie(+movie,+person).\n'),
    file.write('mode: movie(+movie,-person).\n'),
    file.write('mode: movie(-movie,+person).\n'),
    file.write('mode: workedunder(+person,+person).\n'),
    file.write('mode: workedunder(+person,-person).\n'),
    file.write('mode: workedunder(-person,+person).\n'),