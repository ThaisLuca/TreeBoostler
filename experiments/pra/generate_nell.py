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
sys.path.insert(1, os.path.join(sys.path[0], '../../mprob2foil/'))

# Importing the libraries
import pandas as pd
from itertools import product
#import random
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

dataset = pd.read_csv('../../../rule_learning_experiment/NELL.sports.08m.850.small.csv')

ds = get_data(dataset, types)
data = ds[0]
consts = ds[1]

with open('nell/domains.tsv', 'w') as file:
    for i in types:
        file.write(i + '\t' + types[i][0] + '\n')
    
with open('nell/ranges.tsv', 'w') as file:
    for i in types:
        file.write(i + '\t' + types[i][1] + '\n')
        
with open('nell/inverses.tsv', 'w') as file:
    file.write('teamalsoknownas\tteamalsoknownas\n')
    file.write('teamplaysagainstteam\tteamplaysagainstteam\n')
    
with open('nell/labeled_edges.tsv', 'w') as file:
    for i in data:
        for j in data[i]:
            file.write(j[0] + '\t' + j[1] + '\t' + j[2] + '\n')

for i in data:
    with open('nell/relations/'+ i +'.tsv', 'w') as file:
        for j in data[i]:
            file.write(j[0] + '\t' + j[2] + '\n')
            
for i in consts:
    with open('nell/category_instances/'+ i +'.tsv', 'w') as file:
        for j in consts[i]:
            file.write(j + '\n')
    

     
                      