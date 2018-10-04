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
import json
import copy

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class datasets:
    def get_kfold(test_number, folds):
        '''Separate examples into train and test set.
        It uses k-1 folds for training and 1 single fold for testing'''
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
    
    def get_kfold_small(train_number, folds):
        '''Separate examples into train and test set.
        It uses 1 single fold for training and k-1 folds for testing'''
        train = []
        test = []
        for i in range(len(folds)):
            if i == train_number:
                train += folds[i]
            else:
                test += folds[i]
        return (train, test)
    
    def group_folds(folds):
        '''Group folds in a single one'''
        train = []
        for i in range(len(folds)):
            train += folds[i]
        return train
    
    def split_into_folds(examples, n_folds=5):
        '''For datasets as nell and yago that have only 1 mega-example'''
        temp = list(examples)
        s = math.ceil(len(examples)/n_folds)
        ret = []
        for i in range(n_folds-1):
            ret.append(temp[:s])
            temp = temp[s:]
        ret.append(temp)
        return ret
    
    def balance_neg(data, seed=None):
        '''Receives [facts, pos, neg] and balance neg according to pos'''
        facts = copy.deepcopy(data[0])
        pos = copy.deepcopy(data[1])
        neg = copy.deepcopy(data[2])
        random.seed(seed)
        for i in range(len(neg)):
            random.shuffle(neg[i])
            neg[i] = neg[i][:len(pos[i])]
        random.seed(None)
        return [facts, pos, neg]
    
    def generate_neg(data, seed=None):
        '''Receives [facts, pos, neg] and generates balanced neg examples in neg according to pos'''
        facts = copy.deepcopy(data[0])
        pos = copy.deepcopy(data[1])
        neg = copy.deepcopy(data[2])
        pattern = '^(\w+)\(([\w, ]+)*\).$'
        target = None
        for i in range(len(pos)):
            objects = set()
            subjects = {}
            for example in pos[i]:
                m = re.search(pattern, example)
                if m:
                    target = m.group(1)
                    entities = m.group(2).split(',')
                    if entities[0] not in subjects:
                        subjects[entities[0]] = set()
                    subjects[entities[0]].add(entities[1])
                    objects.add(entities[1])
            random.seed(seed)
            target_objects = list(objects)
            for example in pos[i]:
                m = re.search(pattern, example)
                if m:
                    entities = m.group(2).split(',')
                    key = entities[0]
                    for tr in range(10):
                        r = random.randint(0, len(target_objects)-1)
                        if target_objects[r] not in subjects[key]:
                            neg[i].append(target + '(' + ','.join([key, target_objects[r]]) + ').')
                            break
            random.seed(None)
        return [facts, pos, neg]
    
    def target(target, data):
        '''Receives [facts, neg] and returns [facts, pos, neg] with pos and neg containing only target predicates'''
        facts = []
        pos = []
        neg = []
        pattern = '^(\w+)\(([\w, ]+)*\).$'
        for i in range(len(data[0])):
            facts.append([])
            pos.append([])
            for example in data[0][i]:
                m = re.search(pattern, example)
                if m:
                    relation = m.group(1)
                    if relation == target:
                        pos[i].append(example)
                    else:
                        facts[i].append(example)
        for i in range(len(data[1])):
            neg.append([])
            for example in data[1][i]:
                m = re.search(pattern, example)
                if m:
                    relation = m.group(1)
                    if relation == target:
                        neg[i].append(example)
        return [facts, pos, neg]
    
    def get_json_dataset(dataset):
        '''Load dataset from json'''
        with open('files/json/' + dataset + '.json') as data_file:
            data_loaded = json.load(data_file)
        return data_loaded
    
    def load(dataset, bk):
        '''Load dataset from json and accept only predicates presented in bk'''
        pattern = '^(\w+)\(.*\).$'
        accepted = set()
        for line in bk:
            m = re.search(pattern, line)
            if m:
                relation = re.sub('[ _]', '', m.group(1))
                accepted.add(relation)
        data = datasets.get_json_dataset(dataset)
        n_data = [[], []]
        for t in range(len(data)): #positives, negatives
            for i in range(len(data[t])):
                n_data[t].append([])
                for example in data[t][i]:
                    m = re.search(pattern, example)
                    if m:
                        relation = m.group(1)
                        if relation in accepted:
                            n_data[t][i].append(example)
        return n_data

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
    def get_imdb_dataset(acceptedPredicates=None):
        facts = []
        negatives = []
        i = -1
        with open(os.path.join(__location__, 'files/imdb.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line)
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line)
                if b:
                    i += 1
                    facts.append([])
                    negatives.append([])               
                if m:
                    relation = re.sub('[ _]', '', m.group(1))
                    entities = re.sub('[ _]', '', m.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = re.sub('[ _]', '', n.group(1))
                    entities = re.sub('[ _]', '', n.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        return [facts, negatives]

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
    def get_cora_dataset(acceptedPredicates=None):
        facts = []
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
                    negatives.append([])
                if m:
                    relation = re.sub('[ _]', '', m.group(1))
                    entities = re.sub('[ _]', '', m.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = re.sub('[ _]', '', n.group(1))
                    entities = re.sub('[ _]', '', n.group(2)).split(',')
                    if not acceptedPredicates or relation in acceptedPredicates:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        return [facts, negatives]

    '''
    professor(person)
    student(person)
    advisedby(person,person)
    tempadvisedby(person,person)
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
    def get_uwcse_dataset(acceptedPredicates=None):
        facts = []
        negatives = []
        fold = {}
        fold_i = 0
        i = 0
        with open(os.path.join(__location__, 'files/uwcselearn.pl')) as f:
            for line in f:
                n = re.search('^neg\((\w+)\(([\w, ]+)*\)\).$', line) 
                m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
                if m:
                    relation = re.sub('[ _]', '', m.group(1))
                    entities = re.sub('[ _]', '', m.group(2)).split(',')
                    if entities[0] not in fold:
                        fold[entities[0]] = fold_i
                        i = fold_i
                        facts.append([])
                        negatives.append([])
                        fold_i += 1
                    else:
                        i = fold[entities[0]]
                    entities = entities[1:]
                    if not acceptedPredicates or relation in acceptedPredicates:
                        facts[i].append(relation + '(' + ','.join(entities) + ').')
                if n:
                    relation = re.sub('[ _]', '', n.group(1))
                    entities = re.sub('[ _]', '', n.group(2)).split(',')
                    if entities[0] not in fold:
                        fold[entities[0]] = fold_i
                        i = fold_i
                        facts.append([])
                        negatives.append([])
                        fold_i += 1
                    else:
                        i = fold[entities[0]]
                    entities = entities[1:]
                    if not acceptedPredicates or relation in acceptedPredicates:
                        negatives[i].append(relation + '(' + ','.join(entities) + ').')
        return [facts, negatives]
    
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
    def get_webkb_dataset(acceptedPredicates=None):       
        facts = []
        negatives = []
        pages = {}
        count = {'id' : 1}
        i = -1
        
        def getPageId(page):
            if page not in pages:
                pages[page] = 'page' + str(count['id'])
                count['id'] += 1
            return pages[page]
        
        def cleanEntity(entity):
            m = re.search('^(http|https|ftp|mail|file)\:', entity)
            if m:
                return getPageId(entity)
            else:
                return entity
            
        def getCleanEntities(entities):
            new_entities = list(entities)
            return [cleanEntity(entity) for entity in new_entities]
        
        with open(os.path.join(__location__, 'files/webkb.pl')) as f:
            for line in f:
                b = re.search('^begin\(model\([0-9\w]*\)\).$', line.lower())   
                n = re.search('^neg\((\w+)\((.*)\)\).$', line.lower()) 
                m = re.search('^(\w+)\((.*)\).$', line.lower())
                if b:
                    i += 1
                    facts.append([])
                    negatives.append([])
                    continue
                if n:
                    relation = re.sub('[\']', '', n.group(1))
                    if relation not in ['output', 'input_cw', 'input', 'determination', 'begin', 'modeb', 'modeh', 'banned', 'fold', 'lookahead', 'bg', 'in']:
                        entities = re.sub('[\']', '', n.group(2)).split(',')
                        entities = getCleanEntities(entities)
                        if not acceptedPredicates or relation in acceptedPredicates:
                            negatives[i].append(relation + '(' + ','.join(entities) + ').')
                    continue
                if m:
                    relation = re.sub('[\']', '', m.group(1))
                    if relation not in ['output', 'input_cw', 'input', 'determination', 'begin', 'modeb', 'modeh', 'banned', 'fold', 'lookahead', 'bg', 'in']:
                        entities = re.sub('[\']', '', m.group(2)).split(',')
                        entities = getCleanEntities(entities)
                        if not acceptedPredicates or relation in acceptedPredicates:
                            facts[i].append(relation + '(' + ','.join(entities) + ').')
                    continue
        return [facts, negatives]
    
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
    def get_nell_sports_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z]', '', value)
            return value
       
        facts = []
        dataset = pd.read_csv(os.path.join(__location__, 'files/NELL.sports.08m.1070.small.csv'))
        for data in dataset.values:
            entity = clearCharacters((data[1].split(':'))[2])
            relation = clearCharacters((data[4].split(':'))[1])
            value = clearCharacters((data[5].split(':'))[2])
            
            if entity and relation and value:
                if not acceptedPredicates or relation in acceptedPredicates:
                    facts.append(relation + '(' + ','.join([entity, value]) + ').')
        return [[facts], [[]]]
    
    def get_yago2s_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = unidecode.unidecode(value)
            value = re.sub('[^a-z]', '', value)
            return value
        
        facts = []
        with open(os.path.join(__location__, 'files/yago2s.tsv'), encoding='utf8') as f:
            f = csv.reader(f, delimiter='\t')
            for row in f:
                for i in range(len(row)):
                    row[i] = clearCharacters(row[i])
                if row[0] and row[2]:
                    if not acceptedPredicates or row[1] in acceptedPredicates:
                        relation = row[1]
                        entities = [row[0], row[2]]
                        facts.append(relation + '(' + ','.join(entities) + ').')
        return [[facts], [[]]]
    
    '''
    accounttype(account,+type)
    tweets(account,+word)
    follows(account,account)'''  
    def get_twitter_dataset(acceptedPredicates=None):
        facts = [[],[]]
        for i in range(2):
            with open(os.path.join(__location__, 'files/twitter-fold' + str(i+1) + '.db')) as f:
                for line in f:
                    m = re.search('^([\w_]+)\(([\w, "_-]+)*\)$', line.lower())
                    if m:
                        relation = m.group(1)
                        entities = m.group(2)
                        entities = re.sub('[ _"-]', '', entities)
                        entities = entities.split(',')
                        if not acceptedPredicates or relation in acceptedPredicates:
                            facts[i].append(relation + '(' + ','.join(entities) + ').')
        return [facts, [[],[]]]
    
    '''
    location(protein,loc)
    interaction(protein,protein)
    proteinclass(protein,class)
    enzyme(protein,enz)
    function(protein,+fun)
    complex(protein,com)
    phenotype(protein,phe)'''  
    def get_yeast_dataset(acceptedPredicates=None):
        facts = [[],[],[],[]]
        for i in range(4):
            with open(os.path.join(__location__, 'files/yeast-fold' + str(i+1) + '.db')) as f:
                for line in f:
                    m = re.search('^([\w_]+)\(([\w, "_-]+)*\)$', line.lower())
                    if m:
                        relation = m.group(1)
                        relation = re.sub('[_]', '', relation)
                        entities = m.group(2)
                        entities = re.sub('[ _"-]', '', entities)
                        entities = entities.split(',')
                        if not acceptedPredicates or relation in acceptedPredicates:
                            facts[i].append(relation + '(' + ','.join(entities) + ').')
        return [facts, []]
    
    '''
    countryhascompanyoffice(country,company)
    companyeconomicsector(company,sector)
    companyceo(company,person)
    companyalsoknownas(company,company)
    cityhascompanyoffice(city,company)
    '''
    def get_nell_finances_dataset(acceptedPredicates=None):
        def clearCharacters(value):
            value = value.lower()
            value = re.sub('[^a-z]', '', value)
            return value

        facts = []
        dataset = pd.read_csv(os.path.join(__location__, 'files/NELL.finances.08m.1115.small.csv'))
        for data in dataset.values:
            entity = clearCharacters((data[1].split(':'))[2])
            relation = clearCharacters((data[4].split(':'))[1])
            value = clearCharacters((data[5].split(':'))[2])
            
            if entity and relation and value:
                if not acceptedPredicates or relation in acceptedPredicates:
                    facts.append(relation + '(' + ','.join([entity, value]) + ').')
        return [[facts], [[]]]

#import time 
#start = time.time()
#data = datasets.get_yago2s_dataset()
#print(time.time() - start)
#
#import json
#with open('files/json/yago2s.json', 'w') as outfile:
#    json.dump(data, outfile)
        
#import time 
#start = time.time()
#data = datasets.get_json_dataset('uwcse')
#print(time.time() - start) 
#
#start = time.time()
#data2 = datasets.load('uwcse', ['professor(person).',
#    'student(person).',
#    'advisedby(person,person)'
#    'tempadvisedby(person,person).',
#    'hasposition(person,faculty).',
#    'publication(title,person).',
#    'inphase(person, pre_quals).',
#    'courselevel(course,#level).',
#    'yearsinprogram(person,#year).',
#    'projectmember(project, person).'])
#print(time.time() - start) 