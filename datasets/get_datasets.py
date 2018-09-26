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
import math
import random
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class datasets:
    def get_kfold(test_number, folds):
        train = []
        test = []
        for i in range(len(folds)):
            if i == test_number:
                test += folds[i]
            else:
                train += folds[i]
        return (train, test)
    
    def get_kfold_separated(test_number, folds):
        train = []
        test = []
        for i in range(len(folds)):
            if i == test_number:
                test = folds[i]
            else:
                train.append(folds[i])
        return (train, test)
    
    def group_folds(folds):
        train = []
        for i in range(len(folds)):
            train += folds[i]
        return train
    
    def split_into_folds(examples, n_folds=5):
        temp = list(examples)
        s = math.ceil(len(examples)/n_folds)
        ret = []
        for i in range(n_folds-1):
            ret.append(temp[:s])
            temp = temp[s:]
        ret.append(temp)
        return ret
    
    def target_examples(target, data):
        facts = []
        pos = []
        neg = []
        pattern = '^(\w+)\(([\w, ]+)*\).$'
        for i in range(len(data[1])):
            facts.append([])
            pos.append([])
            for example in data[1][i]:
                m = re.search(pattern, example)
                if m:
                    relation = m.group(1).replace(' ', '')
                    if relation == target:
                        pos[i].append(example)
                    else:
                        facts[i].append(example)
        for i in range(len(data[2])):
            neg.append([])
            for example in data[2][i]:
                m = re.search(pattern, example)
                if m:
                    relation = m.group(1).replace(' ', '')
                    if relation == target:
                        neg[i].append(example)
        return [facts, pos, neg]
                        
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
    def get_imdb_dataset(target, acceptedPredicates=None, folds=False):
        facts = []
        positives = []
        negatives = []
        i = -1
        with open(os.path.join(__location__, 'files/imdb.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line)
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                if b:
                    i += 1
                    facts.append([])
                    positives.append([])
                    negatives.append([])
                if m:
                    relation = m.group(1).replace(' ', '')
                    entities = m.group(2).replace(' ', '').replace('_','').split(',')
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        positives[i].append(relation + '(' + ','.join(entities) + ').')
                    elif not acceptedPredicates or relation in acceptedPredicates:
                        facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = n.group(1).replace(' ', '')
                    entities = n.group(2).replace(' ', '').replace('_','').split(',')
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        if folds:
            return [facts, positives, negatives]
        else:
            return [[j for i in facts for j in i], [j for i in positives for j in i], [j for i in negatives for j in i]]

    '''
    samebib(class,class)
    sameauthor(author,author)
    sametitle(title,title)
    samevenue(venue,venue)
    author(class,author)
    title(class,title)
    venue(class,venue)
    haswordauthor(author,word)
    harswordtitle(title,word)
    haswordvenue(venue,word)
    '''  
    def get_cora_dataset(target, acceptedPredicates=None, folds=False):
        facts = []
        positives = []
        negatives = []
        i = -1
        with open(os.path.join(__location__, 'files/coralearn.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line)
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                if b:
                    i += 1
                    facts.append([])
                    positives.append([])
                    negatives.append([])
                if m:
                    relation = m.group(1).replace(' ', '')
                    entities = m.group(2).replace(' ', '').replace('_','').split(',')
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        positives[i].append(relation + '(' + ','.join(entities) + ').')
                    elif not acceptedPredicates or relation in acceptedPredicates:
                        facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = n.group(1).replace(' ', '')
                    entities = n.group(2).replace(' ', '').replace('_','').split(',')
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        if folds:
            return [facts, positives, negatives]
        else:
            return [[j for i in facts for j in i], [j for i in positives for j in i], [j for i in negatives for j in i]]

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
    def get_uwcse_dataset(target, acceptedPredicates=None, folds=False):
        facts = []
        positives = []
        negatives = []
        fold = {}
        fold_i = 0
        i = 0
        with open(os.path.join(__location__, 'files/uwcselearn.pl')) as f:
            for line in f:
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)               
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                if m:
                    relation = m.group(1).replace(' ', '')
                    entities = m.group(2).replace(' ', '').replace('_','').split(',')
                    if entities[0] not in fold:
                        fold[entities[0]] = fold_i
                        i = fold_i
                        positives.append([])
                        facts.append([])
                        negatives.append([])
                        fold_i += 1
                    else:
                        i = fold[entities[0]]
                    entities= entities[1:]
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        positives[i].append(relation + '(' + ','.join(entities) + ').')
                    elif not acceptedPredicates or relation in acceptedPredicates:
                        facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = n.group(1).replace(' ', '')
                    entities = n.group(2).replace(' ', '').replace('_','').split(',')
                    if entities[0] not in fold:
                        fold[entities[0]] = fold_i
                        i = fold_i
                        positives.append([])
                        facts.append([])
                        negatives.append([])
                        fold_i += 1
                    else:
                        i = fold[entities[0]]
                    entities= entities[1:]
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        if folds:
            return [facts, positives, negatives]
        else:
            return [[j for i in facts for j in i], [j for i in positives for j in i], [j for i in negatives for j in i]]
    
    '''
    coursePage(page)
    facultyPage(page)
    studentPage(page)
    linkTo(id,page,page)
    researchProjectPage(page)
    has(word,page)
    hasAlphanumericWord(id)
    allWordsCapitalized(id)
    '''  
    def get_webkb_dataset(target, acceptedPredicates=None, folds=False):
        facts = []
        positives = []
        negatives = []
        i = -1
        with open(os.path.join(__location__, 'files/webkb.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line)
                n = re.search('^neg\((\w+)\(([\w, \'\:\/\.]+)*\)\).$', line)               
                m = re.search('^(\w+)\(([\w, \'\:\/\.]+)*\).$', line)
                if b:
                    i += 1
                    facts.append([])
                    positives.append([])
                    negatives.append([])
                if m:
                    relation = re.sub('(http\:\/\/)|(www)|[ _\'\:\.\/]', '', m.group(1))
                    if relation not in ['output', 'inputcw', 'input', 'determination']:
                        entities = re.sub('(http\:\/\/)|(www)|[ _\'\:\.\/]', '', m.group(2)).split(',')
                        if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                            positives[i].append(relation + '(' + ','.join(entities) + ').')
                        elif not acceptedPredicates or relation in acceptedPredicates:
                            facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = n.group(1).replace(' ', '')
                    entities = n.group(2).replace(' ', '').replace('_','').split(',')
                    if (not target and (not acceptedPredicates or relation in acceptedPredicates)) or relation == target:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        if folds:
            return [facts, positives, negatives]
        else:
            return [[j for i in facts for j in i], [j for i in positives for j in i], [j for i in negatives for j in i]]
    
    
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
    def get_nell_dataset(target, acceptedPredicates=None, seed=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z]', '', value)
            return value
    
        random.seed(seed)
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
                if (not target and (not acceptedPredicates or relation in acceptedPredicates)):
                    positives.append(relation + '(' + ','.join([entity, value]) + ').')
                elif relation == target:
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
        random.seed(None)
        return [facts, positives, negatives]
    
    def get_yago2s_dataset(target, acceptedPredicates=None, seed=None):
        def clearCharacters(value):
            value = value.lower()
            value = unidecode.unidecode(value)
            value = re.sub('[^a-z]', '', value)
            return value
        
        random.seed(seed)
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
        random.seed(None)
        return [facts, positives, negatives]
