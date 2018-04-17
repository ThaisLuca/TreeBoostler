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

# add the parent folder to python path
import os, sys

# Importing the libraries
import pandas as pd
from itertools import product
import random
import re
#import time

types = {
'athleteledsportsteam': ('athlete', 'sportsteam'),
'athleteplaysforteam': ('athlete', 'sportsteam'),
'athleteplaysinleague': ('athlete', 'sportsleague'),
'athleteplayssport': ('athlete', 'sport'),
'teamalsoknownas': ('sportsteam', 'sportsteam'),
'teamplaysagainstteam': ('sportsteam', 'sportsteam'),
'teamplaysinleague': ('sportsteam', 'sportsleague'),
'teamplayssport': ('sportsteam', 'sport'),
}

target = 'athleteplaysforteam'
n_folds = 5

def create_folds(data, size):
    length = int(len(data)/size) #length of each fold
    folds = []
    for i in range((size-1)):
        folds += [data[i*length:(i+1)*length]]
    folds += [data[(size-1)*length:len(data)]]
    return folds

def get_data(dataset, types):   
    relations = {}
    consts = {}
    for data in dataset.values:
        entity = (data[1].split(':'))[2]
        relation = (data[4].split(':'))[1]
        value = (data[5].split(':'))[2]
           
        entity = entity.lower() #.replace('_', '')
        value = value.lower() #.replace('_', '')
        #re.sub('[^a-zA-Z]', '', title[j])
        entity = re.sub('[^a-z_]', '', entity)
        value = re.sub('[^a-z_]', '', value)
        
        #entity and value cannot start with '_', otherwise it is considered variable (?)
        entity = entity[1:] if entity[0] == '_' else entity
        value = value[1:] if value[0] == '_' else value
                  
        if relation in relations:
            relations[relation].append([entity, relation, value])
        else:
            relations[relation] = [[entity, relation, value]]

        sub = types[relation][0]
        obj = types[relation][1]

        if sub in consts:
            consts[sub].add(entity)
        else:
            consts[sub] = set([entity])
            
        if obj in consts:
            consts[obj].add(value)
        else:
            consts[obj] = set([value])

    return (relations, consts)

dataset = pd.read_csv('/home/rodrigoazs/Projetos/rule_learning_experiment/NELL.sports.08m.850.small.csv')

ds = get_data(dataset, types)
data = ds[0]
consts = ds[1]

tar = data[target]
values = [set(), set()]
for d in tar:
    values[0].add(str(d[0]))
    values[1].add(str(d[2]))

#random.shuffle(tar)
#tar = create_folds(tar, n_folds) 
pos_count = len(tar)
random.shuffle(tar)
tar = create_folds(tar, n_folds)
neg_examples = list(product(*values))
random.shuffle(neg_examples)
neg_examples = neg_examples[:pos_count]
neg_examples = create_folds(neg_examples, n_folds)

train_pos = open('nell/train/train_pos.txt', 'w')
train_neg = open('nell/train/train_neg.txt', 'w')
test_pos = open('nell/test/test_pos.txt', 'w')
test_neg = open('nell/test/test_neg.txt', 'w')

for i in range(n_folds-1):
    # print positive and negative targets
    for j in range(len(tar[i])):
        d = tar[i][j]
        train_pos.write(str(d[1]) + '(' +str(d[0])+ ','+str(d[2])+ ').\n')
        train_neg.write(str(d[1]) + '(' +str(neg_examples[i][j][0])+ ','+ neg_examples[i][j][1] + ').\n')

i = n_folds - 1
for j in range(len(tar[i])):
    d = tar[i][j]
    test_pos.write(str(d[1]) + '(' +str(d[0])+ ','+str(d[2])+ ').\n')
    test_neg.write(str(d[1]) + '(' +str(neg_examples[i][j][0])+ ','+ neg_examples[i][j][1] + ').\n')   

train_facts = open('nell/train/train_facts.txt', 'w')
test_facts = open('nell/test/test_facts.txt', 'w')
for i in data:
    if i != target:
        for j in data[i]:
            train_facts.write(j[1] + '(' + j[0] + ',' + j[2] + ').\n')
            test_facts.write(j[1] + '(' + j[0] + ',' + j[2] + ').\n')
            
with open('nell/train/train_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')
    
with open('nell/test/test_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')

with open('nell/background.txt', 'w') as file:    
    file.write('//Parameters\n')
    file.write('setParam: maxTreeDepth=3.\n')
    file.write('setParam: nodeSize=1.\n')
    file.write('setParam: numOfClauses=8.\n')
    file.write('//Modes\n')
    file.write('mode: male(+name).\n')
    file.write('mode: athleteledsportsteam(+athlete,+sportsteam).\n'),
    file.write('mode: athleteledsportsteam(-athlete,+sportsteam).\n'),
    file.write('mode: athleteledsportsteam(+athlete,-sportsteam).\n'),
    file.write('mode: athleteplaysforteam(+athlete,+sportsteam).\n'),
    file.write('mode: athleteplaysforteam(+athlete,-sportsteam).\n'),
    file.write('mode: athleteplaysforteam(-athlete,+sportsteam).\n'),
    file.write('mode: athleteplaysinleague(+athlete,+sportsleague).\n'),
    file.write('mode: athleteplaysinleague(+athlete,-sportsleague).\n'),
    file.write('mode: athleteplaysinleague(-athlete,+sportsleague).\n'),
    file.write('mode: athleteplayssport(+athlete,+sport).\n'),
    file.write('mode: athleteplayssport(+athlete,-sport).\n'),
    file.write('mode: athleteplayssport(-athlete,+sport).\n'),
    file.write('mode: teamalsoknownas(+sportsteam,+sportsteam).\n'),
    file.write('mode: teamalsoknownas(+sportsteam,-sportsteam).\n'),
    file.write('mode: teamalsoknownas(-sportsteam,+sportsteam).\n'),
    file.write('mode: teamplaysagainstteam(+sportsteam,+sportsteam).\n'),
    file.write('mode: teamplaysagainstteam(+sportsteam,-sportsteam).\n'),
    file.write('mode: teamplaysagainstteam(-sportsteam,+sportsteam).\n'),
    file.write('mode: teamplaysinleague(+sportsteam,+sportsleague).\n'),
    file.write('mode: teamplaysinleague(+sportsteam,-sportsleague).\n'),
    file.write('mode: teamplaysinleague(-sportsteam,+sportsleague).\n'),
    file.write('mode: teamplayssport(+sportsteam,+sport).\n'),
    file.write('mode: teamplayssport(+sportsteam,-sport).\n'),
    file.write('mode: teamplayssport(-sportsteam,+sport).\n'),
    file.write('//Bridgers\n')
    file.write('//bridger: siblingof/2.\n')