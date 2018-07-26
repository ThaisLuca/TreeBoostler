'''
   Functions to return datasets in file folder
   Name:         get_datasets.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

import re
import os
import unidecode
import csv
import random
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

'''
workedunder(person,person)
genre(person,genre)
female(person)
actor(person)
director(person)
movie(movie,person)
genre(person,genre)
genre(person,ascifi)
genre(person,athriller)
genre(person,adrama)
genre(person,acrime)
genre(person,acomedy)
genre(person,amystery)
genre(person,aromance)'''  
def get_imdb_dataset(target, acceptedPredicates=None):
    facts = []
    positives = []
    negatives = []
    with open(os.path.join(__location__, 'files/imdb.pl')) as f:
        for line in f:
            n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
            m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
            if m:
                relation = m.group(1).replace(' ', '')
                entities = m.group(2).replace(' ', '').replace('_','').split(',')
                if relation == target:
                    positives.append(relation + '(' + ','.join(entities) + ').')
                elif not acceptedPredicates or relation in acceptedPredicates:
                    facts.append(relation + '(' + ','.join(entities) + ').')
            if n:
                relation = n.group(1).replace(' ', '')
                entities = n.group(2).replace(' ', '').replace('_','').split(',')
                if relation == target:
                    negatives.append(relation + '(' + ','.join(entities) + ').')
    return [facts, positives, negatives]

'''
professor(person)
student(person)
advisedby(person,person)
tempadvisedby(person,person)
taughtby(course,person)
ta(course,person,quarter)
hasposition(person,faculty)
publication(title,person)
inphase(person, pre_quals)
taughtby(course, person, quarter)
courselevel(course,#level)
yearsinprogram(person,#year)
projectmember(project, person)
sameproject(project, project)
samecourse(course, course)
sameperson(person, person)'''                  
def get_uwcse_dataset(target, acceptedPredicates=None):
    facts = []
    positives = []
    negatives = []
    with open(os.path.join(__location__, 'files/uwcselearn.pl')) as f:
        for line in f:
            n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
            m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
            if m:
                relation = m.group(1).replace(' ', '')
                entities = m.group(2).replace(' ', '').replace('_','').split(',')
                entities= entities[1:]
                if relation == target:
                    positives.append(relation + '(' + ','.join(entities) + ').')
                elif not acceptedPredicates or relation in acceptedPredicates:
                    facts.append(relation + '(' + ','.join(entities) + ').')
            if n:
                relation = n.group(1).replace(' ', '')
                entities = n.group(2).replace(' ', '').replace('_','').split(',')
                entities= entities[1:]
                if relation == target:
                    negatives.append(relation + '(' + ','.join(entities) + ').')
    return [facts, positives, negatives]

'''
athleteledsportsteam(athlete,sportsteam)
athleteplaysforteam(athlete,sportsteam)
athleteplaysinleague(athlete,sportsleague)
athleteplayssport(athlete,sport)
teamalsoknownas(sportsteam,sportsteam)
teamplaysagainstteam(sportsteam,sportsteam)
teamplaysinleague(sportsteam,sportsleague)
teamplayssport(sportsteam,sport)
'''
def get_nell_dataset(target, acceptedPredicates=None):
    def clearCharacters(value):
        value = value.lower()
        value = re.sub('[^a-z]', '', value)
        return value

    target_tuples = []
    target_subjects = {}
    target_objects = set()
    
    facts = []
    positives = []
    negatives = []
    dataset = pd.read_csv(os.path.join(__location__, 'files/NELL.sports.08m.1070.small.csv'))
    for data in dataset.values:
        entity = clearCharacters((data[1].split(':'))[2])
        relation = clearCharacters((data[4].split(':'))[1])
        value = clearCharacters((data[5].split(':'))[2])
        
        if entity and relation and value:
            if relation == target:
                target_tuples.append((entity, value))
                if entity in target_subjects:
                    target_subjects[entity].append(value)
                else:
                    target_subjects[entity] = [value]
                target_objects.add(value)
                positives.append(relation + '(' + ','.join([entity, value]) + ').')
            elif not acceptedPredicates or relation in acceptedPredicates:
                    facts.append(relation + '(' + ','.join([entity, value]) + ').')
    target_objects = list(target_objects)
    for tupl in target_tuples:
        i = random.randint(0, len(target_objects)-1)
        while target_objects[i] in target_subjects[tupl[0]]:
            i = random.randint(0, len(target_objects)-1)
        entities = [tupl[0], target_objects[i]]
        relation = target
        negatives.append(relation + '(' + ','.join(entities) + ').')
    return [facts, positives, negatives]

def get_yago2s_dataset(target, acceptedPredicates=None):
    def clearCharacters(value):
        value = value.lower()
        value = unidecode.unidecode(value)
        value = re.sub('[^a-z]', '', value)
        return value
        
    target_tuples = []
    target_subjects = {}
    target_objects = set()
    
    facts = []
    positives = []
    negatives = []
    with open(os.path.join(__location__, 'files/yago2s.tsv')) as f:
        f = csv.reader(f, delimiter='\t')
        for row in f:
            for i in range(len(row)):
                row[i] = clearCharacters(row[i])
            if row[0] and row[2]:
                if row[1] == target:
                    if row[1] == 'hasgender':
                        relation = row[2]
                        entities = [row[0]]
                        target_tuples.append((row[0], row[2]))
                        if row[0] in target_subjects:
                            target_subjects[row[0]].append(row[1])
                        else:
                            target_subjects[row[0]] = [row[1]]
                        target_objects.add(row[1])
                    else:
                        relation = row[1]
                        entities = [row[0], row[2]]
                        target_tuples.append((row[0], row[2]))
                        if row[0] in target_subjects:
                            target_subjects[row[0]].append(row[2])
                        else:
                            target_subjects[row[0]] = [row[2]]
                        target_objects.add(row[2])
                    positives.append(relation + '(' + ','.join(entities) + ').')
                elif not acceptedPredicates or row[1] in acceptedPredicates:
                    relation = row[1]
                    entities = [row[0], row[2]]
                    facts.append(relation + '(' + ','.join(entities) + ').')
    target_objects = list(target_objects)
    for tupl in target_tuples:
        i = random.randint(0, len(target_objects)-1)
        while target_objects[i] in target_subjects[tupl[0]]:
            i = random.randint(0, len(target_objects)-1)
        if target == 'hasgender':
            entities = [tupl[0]]
            relation = target_objects[i]
        else:
            entities = [tupl[0],target_objects[i]]
            relation = target
        negatives.append(relation + '(' + ','.join(entities) + ').')
    return [facts, positives, negatives]
