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
'female': ('person', 'person'),
'actor': ('person', 'person'),
'director': ('person', 'person'),
'movie': ('movie', 'person'),
#'isa': ('person','type')
}

def get_data(types):
    relations = {}
    consts = {}
    with open('/home/rodrigoazs/Projetos/rule_learning_experiment/slipcover/others/imdb.pl') as f:
        for line in f:
            m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
            if m:
                relation = m.group(1).replace(' ', '')
                entities = m.group(2).replace(' ', '').split(',')
                entity = entities[0]
                value = entities[1] if len(entities) > 1 else entities[0] #relation
                #relation = relation if len(entities) > 1 else 'isa'
               
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

ds = get_data(types)
data = ds[0]
consts = ds[1]

with open('imdb/domains.tsv', 'w') as file:
    for i in types:
        file.write(i + '\t' + types[i][0] + '\n')
    
with open('imdb/ranges.tsv', 'w') as file:
    for i in types:
        file.write(i + '\t' + types[i][1] + '\n')
        
with open('imdb/inverses.tsv', 'w') as file:
    file.write('director\tdirector\n')
    file.write('actor\tactor\n')
    file.write('female\tfemale\n')
    
with open('imdb/labeled_edges.tsv', 'w') as file:
    for i in data:
        for j in data[i]:
            file.write(j[0] + '\t' + j[1] + '\t' + j[2] + '\n')

os.makedirs('imdb/relations', exist_ok=True)
for i in data:
    with open('imdb/relations/'+ i, 'w') as file:
        for j in data[i]:
            file.write(j[0] + '\t' + j[2] + '\n')

os.makedirs('imdb/category_instances', exist_ok=True)
for i in consts:
    with open('imdb/category_instances/'+ i, 'w') as file:
        for j in consts[i]:
            file.write(j + '\n')
    

     
                      