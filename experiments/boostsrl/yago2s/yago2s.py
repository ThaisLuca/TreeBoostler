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
 
    
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

test_examples_proportion = 0.2
target = 'playsfor'
ignore = ['haswebsite', 'isconnectedto', 'hasgender', 'islocatedin']
target_tuples = []
target_subjects = {}
target_objects = set()
predicates = set()

start_time = time.time()

with open('yago2s.tsv','r') as tsvin, open('train/train_facts.txt', 'w') as out:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tsvin:
        for i in range(len(row)):
            row[i] = clearCharacters(row[i])
        # subjects and objects that starts with _ are not accepted
        if row[1] not in ignore and row[0] and row[2] and row[0][:1] != '_' and row[2][:1] != '_':
            predicates.add(row[1])
            if row[1] == target:
                #pos.write(row[1] + '(' + row[0] + ',' + row[2] + ').\n')
                target_tuples.append((row[0], row[2]))
                if row[0] in target_subjects:
                    target_subjects[row[0]].append(row[2])
                else:
                    target_subjects[row[0]] = [row[2]]
                target_objects.add(row[2])
            else:
                if row[1] == 'hasgender':
                    out.write(row[2] + '(' + row[0] + ').\n')
                else:
                    out.write(row[1] + '(' + row[0] + ',' + row[2] + ').\n')

# copy train_facts.txt to test_facts.txt
copyfile('train/train_facts.txt', 'test/test_facts.txt')

with open('train/train_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')
    
with open('test/test_bk.txt', 'w') as file:
    file.write('import: "../background.txt".')
                    
with open('background.txt', 'w') as file:    
    file.write('//Parameters\n')
    file.write('setParam: maxTreeDepth=8.\n')
    file.write('setParam: nodeSize=3.\n')
    file.write('setParam: numOfClauses=8.\n')
    file.write('//Modes\n')
    #file.write('mode: male(+person).\n')
    #file.write('mode: female(+person).\n')
    file.write('mode: playsfor(+person,+team).\n')
    file.write('mode: playsfor(+person,-team).\n')
    file.write('mode: playsfor(-person,+team).\n')
    #for predicate in list(predicates):
    file.write('mode: hascurrency(+place,+currency).\n')
    file.write('mode: hascurrency(+place,-currency).\n')
    file.write('mode: hascurrency(-place,+currency).\n')
    file.write('mode: hascapital(+place,+place).\n')
    file.write('mode: hascapital(+place,-place).\n')
    file.write('mode: hascapital(-place,+place).\n')
    file.write('mode: hasacademicadvisor(+person,+person).\n')
    file.write('mode: hasacademicadvisor(+person,-person).\n')
    file.write('mode: hasacademicadvisor(-person,+person).\n')
    file.write('mode: haswonprize(+person,+prize).\n')
    file.write('mode: haswonprize(+person,-prize).\n')
    file.write('mode: haswonprize(-person,+prize).\n')
    file.write('mode: participatedin(+place,+event).\n')
    file.write('mode: participatedin(+place,-event).\n')
    file.write('mode: participatedin(-place,+event).\n')
    file.write('mode: owns(+institution,+institution).\n')
    file.write('mode: owns(+institution,-institution).\n')
    file.write('mode: owns(-institution,+institution).\n')
    file.write('mode: isinterestedin(+person,+concept).\n')
    file.write('mode: isinterestedin(+person,-concept).\n')
    file.write('mode: isinterestedin(-person,+concept).\n')
    file.write('mode: livesin(+person,+place).\n')
    file.write('mode: livesin(+person,-place).\n')
    file.write('mode: livesin(-person,+place).\n')
    file.write('mode: happenedin(+event,+place).\n')
    file.write('mode: happenedin(+event,-place).\n')
    file.write('mode: happenedin(-event,+place).\n')
    file.write('mode: holdspoliticalposition(+person,+political_position).\n')
    file.write('mode: holdspoliticalposition(+person,-political_position).\n')
    file.write('mode: holdspoliticalposition(-person,+political_position).\n')
    #file.write('mode: islocatedin(+place,+place).\n')
    #file.write('mode: islocatedin(+place,-place).\n')
    #file.write('mode: islocatedin(-place,+place).\n')
    file.write('mode: diedin(+person,+place).\n')
    file.write('mode: diedin(+person,-place).\n')
    file.write('mode: diedin(-person,+place).\n')
    file.write('mode: actedin(+person,+media).\n')
    file.write('mode: actedin(+person,-media).\n')
    file.write('mode: actedin(-person,+media).\n')
    file.write('mode: iscitizenof(+person,+place).\n')
    file.write('mode: iscitizenof(+person,-place).\n')
    file.write('mode: iscitizenof(-person,+place).\n')
    file.write('mode: worksat(+person,+institution).\n')
    file.write('mode: worksat(+person,-institution).\n')
    file.write('mode: worksat(-person,+institution).\n')
    file.write('mode: directed(+person,+media).\n')
    file.write('mode: directed(+person,-media).\n')
    file.write('mode: directed(-person,+media).\n')
    file.write('mode: dealswith(+place,+place).\n')
    file.write('mode: dealswith(+place,-place).\n')
    file.write('mode: dealswith(-place,+place).\n')
    file.write('mode: wasbornin(+person,+place).\n')
    file.write('mode: wasbornin(+person,-place).\n')
    file.write('mode: wasbornin(-person,+place).\n')
    file.write('mode: created(+person,+media).\n')
    file.write('mode: created(+person,-media).\n')
    file.write('mode: created(-person,+media).\n')
    file.write('mode: isleaderof(+person,+place).\n')
    file.write('mode: isleaderof(+person,-place).\n')
    file.write('mode: isleaderof(-person,+place).\n')
    file.write('mode: haschild(+person,+person).\n')
    file.write('mode: haschild(+person,-person).\n')
    file.write('mode: haschild(-person,+person).\n')
    file.write('mode: ismarriedto(+person,+person).\n')
    file.write('mode: ismarriedto(+person,-person).\n')
    file.write('mode: ismarriedto(-person,+person).\n')
    file.write('mode: imports(+person,+material).\n')
    file.write('mode: imports(+person,-material).\n')
    file.write('mode: imports(-person,+material).\n')
    file.write('mode: hasmusicalrole(+person,+musical_role).\n')
    file.write('mode: hasmusicalrole(+person,-musical_role).\n')
    file.write('mode: hasmusicalrole(-person,+musical_role).\n')
    #file.write('mode: isconnectedto(+place,+place).\n')
    #file.write('mode: isconnectedto(+place,-place).\n')
    #file.write('mode: isconnectedto(-place,+place).\n')
    file.write('mode: influences(+person,+person).\n')
    file.write('mode: influences(+person,-person).\n')
    file.write('mode: influences(-person,+person).\n')
    file.write('mode: isaffiliatedto(+person,+team).\n')
    file.write('mode: isaffiliatedto(+person,-team).\n')
    file.write('mode: isaffiliatedto(-person,+team).\n')
    file.write('mode: isknownfor(+person,+theory).\n')
    file.write('mode: isknownfor(+person,-theory).\n')
    file.write('mode: isknownfor(-person,+theory).\n')
    file.write('mode: ispoliticianof(+person,+place).\n')
    file.write('mode: ispoliticianof(+person,-place).\n')
    file.write('mode: ispoliticianof(-person,+place).\n')
    file.write('mode: graduatedfrom(+person,+institution).\n')
    file.write('mode: graduatedfrom(+person,-institution).\n')
    file.write('mode: graduatedfrom(-person,+institution).\n')
    file.write('mode: exports(+place,+material).\n')
    file.write('mode: exports(+place,-material).\n')
    file.write('mode: exports(-place,+material).\n')
    file.write('mode: edited(+person,+media).\n')
    file.write('mode: edited(+person,-media).\n')
    file.write('mode: edited(-person,+media).\n')
    file.write('mode: wrotemusicfor(+person,+media).\n')
    file.write('mode: wrotemusicfor(+person,-media).\n')
    file.write('mode: wrotemusicfor(-person,+media).\n')
        
random.shuffle(target_tuples)
n = int(test_examples_proportion*len(target_tuples))
test = target_tuples[:n]
train = target_tuples[n:]

newset = list(target_objects)
random.shuffle(newset)
with open('train/train_pos.txt', 'w') as pos, open('train/train_neg.txt', 'w') as neg:
    for row in train:
        pos.write(target + '(' + row[0] + ',' + row[1] + ').\n')
        i = random.randint(0, len(newset)-1)
        while newset[i] in target_subjects[row[0]]:
            i = random.randint(0, len(newset)-1)
        neg.write(target + '(' + row[0] + ',' + newset[i] + ').\n')
        
with open('test/test_pos.txt', 'w') as pos, open('test/test_neg.txt', 'w') as neg:
    for row in test:
        pos.write(target + '(' + row[0] + ',' + row[1] + ').\n')
        i = random.randint(0, len(newset)-1)
        while newset[i] in target_subjects[row[0]]:
            i = random.randint(0, len(newset)-1)
        neg.write(target + '(' + row[0] + ',' + newset[i] + ').\n')

elapsed_time = time.time() - start_time
print('Execution time: %.3f' % (elapsed_time))