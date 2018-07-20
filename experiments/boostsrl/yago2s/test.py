#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 20:13:02 2018

@author: rodrigoazs
"""

import time
import csv
import re
import unidecode
import random
import os
from shutil import copyfile
#from itertools import product

def clearCharacters(value):
    value = value.lower()
    value = unidecode.unidecode(value)
    value = re.sub('[^a-z_]', '', value)
    return value
 
class Node:
    edges = []

    def __init__(self, name):
        self.name = name

    def add_edge(self, predicate, obj):
        self.edges.append((predicate, obj))
        
class Graph:
    nodes = {}
    print_relations = set()
    
    def add_relation(self, sub, predicate, obj):
        if sub not in self.nodes:
            self.nodes[sub] = Node(sub)
        if obj not in self.nodes:
            self.nodes[obj] = Node(obj)
        s = self.nodes[sub]
        o = self.nodes[obj]

        s.add_edge(predicate, o)
        o.add_edge('_' + predicate, s)
        
    def get_relations(self, root, depth, maxDepth = 4):
        if depth < maxDepth:
            for edge in root.edges:
                self.print_relations.add(edge[0] + '(' + root.name +', ' + edge[1].name + ').')
            self.get_relations(edge[1], depth+1, maxDepth)

#os.makedirs('train', exist_ok=True)
#os.makedirs('test', exist_ok=True)

test_examples_proportion = 0.1
target = 'playsfor'
ignore = ['haswebsite']
target_tuples = []
target_objects = set()
predicates = set()

entities = {}
target_entities = set()

#playsfor (person, team
#hasgender -> male() and female()
#haschild -> (person, person)
#happenedin (event, place)
#owns could be place, university, person, ...

'''
{'hascurrency' (country, state, place, ...) (currency)
, 'hascapital' (place, country, state) (place, capital)
, 'hasacademicadvisor' (person) (person)
, 'haswonprize' (person) (prize)
, 'participatedin' (country, place, empire) (event, battle)
, 'hasofficiallanguage' (country, state, place) (language)
, 'owns' (company, university) (company, tv channel,...)
, 'isinterestedin' (person) (thoery, ideology, reigion, ...)
, 'livesin' (person) (place, country, state, city)
, 'hasgender'
, 'happenedin' (event, battle) (place, country)
, 'holdspoliticalposition' (person) (political_position)
, 'islocatedin' (place, volcano, mountain, park) (place, state, country)
, 'playsfor' (person) (team)
, 'diedin' (person) (place)
, 'actedin' (person) (tv series, movie)
, 'iscitizenof' (person) (country)
, 'worksat' (person) (university, organization)
, 'directed' (person) (tv series, movie)
, 'dealswith' (country) (country)
, 'wasbornin' (person) (city)
, 'created' (person) (song, album, ??? need recheck)
, 'isleaderof' (person) (city, country, institute)
, 'haschild' (person) (person)
, 'ismarriedto' (person) (person)
, 'imports' (country) (material)
, 'hasmusicalrole' (person) (musical_role)
, 'isconnectedto' (aiport, place) (airport, place)
, 'influences' (person) (person)
, 'isaffiliatedto' (person) (club, team, political party)
, 'isknownfor' (person) (book, ideology, theory)
, 'ispoliticianof' (person) (city, place)
, 'graduatedfrom' (person) (university)
, 'exports' (place, country) (material)
, 'edited' (person) (movie, tv serie)
, 'wrotemusicfor' (person) (movie, tv serie)
'''

start_time = time.time()


graph = Graph()

with open('yago2s.tsv','r') as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tsvin:
        for i in range(len(row)):
            row[i] = clearCharacters(row[i])
        if row[1] and row[0] and row[2] and row[0][:1] != '_' and row[2][:1] != '_':
            if row[0] in entities:
                entities[row[0]].append((0, row[1]))
            else:
                entities[row[0]] = [(0, row[1])]
            if row[2] in entities:
                entities[row[2]].append((1, row[2]))
            else:
                entities[row[2]] = [(1, row[2])]

            if row[1] == target:
                target_entities.add(row[0])
                target_entities.add(row[2])

            graph.add_relation(row[0], row[1], row[2])

# appears once
once = []
for entity in entities.items():
    if len(entity[1]) == 1:
        once.append(entity)

l = list(target_entities)
random.shuffle(l)
l = l[:100]

for e in l:
    graph.get_relations(graph.nodes[e], 0, maxDepth = 1)

print(str(len(once)) + ' entities appears only once.')


elapsed_time = time.time() - start_time
print('Execution time: %.3f' % (elapsed_time))