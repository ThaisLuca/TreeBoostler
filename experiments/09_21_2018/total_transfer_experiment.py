'''
   Learning curve experiment
   Name:         learning_curve.py
   Author:       Rodrigo Azevedo
   Updated:      July 23, 2018
   License:      GPLv3
'''

import os
import sys
import time
#sys.path.append('../..')

from datasets.get_datasets import *
from boostedrevision import *
from boostedtransfer import *
from boostsrl import boostsrl
from sklearn.model_selection import KFold
from mapping import *
import numpy as np
import random
import json
import math

def split_into_folds(examples, n_folds=5):
    temp = list(examples)
    random.shuffle(temp)
    s = math.ceil(len(examples)/n_folds)
    ret = []
    for i in range(n_folds-1):
        ret.append(temp[:s])
        temp = temp[s:]
    ret.append(temp)
    return ret
    
firstRun = False
n_runs = 5
verbose = True

if os.path.isfile('total_transfer_experiment.json'):
    with open('total_transfer_experiment.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': {}}
    firstRun = True

def save(data):
    with open('total_transfer_experiment.json', 'w') as fp:
        json.dump(data, fp)
        
if firstRun:
    results['save'] = {'experiment': 0, 'run': 0 }

imdb_target = 'workedunder'
[imdb_facts, imdb_pos, imdb_neg] = get_imdb_dataset(imdb_target, folds=True)
imdb_bk = ['workedunder(+person,+person).',
          'workedunder(+person,-person).',
          'workedunder(-person,+person).',
          'female(+person).',
          'actor(+person).',
          'director(+person).',
          'movie(+movie,+person).',
          'movie(+movie,-person).',
          'movie(-movie,+person).',
          'genre(+person,+genre).']
uwcse_target = 'advisedby'
[uwcse_facts, uwcse_pos, uwcse_neg] = get_uwcse_dataset(uwcse_target, folds=True, acceptedPredicates=[
    'professor',
    'student',
    'advisedby',
    'tempadvisedby',
    'ta',
    'hasposition',
    'publication',
    'inphase',
    'courselevel',
    'yearsinprogram',
    'projectmember',
    ])
uwcse_bk = ['professor(+person).',
    'student(+person).',
    'advisedby(+person,+person).',
    'advisedby(+person,-person).',
    'advisedby(-person,+person).',
    'tempadvisedby(+person,+person).',
    'tempadvisedby(+person,-person).',
    'tempadvisedby(-person,+person).',
    #'ta(+course,+person,+quarter).',
    #'ta(-course,+person,+quarter).',
    #'ta(+course,-person,+quarter).',
    #'ta(+course,+person,-quarter).',
    #'ta(-course,+person,-quarter).',
    #'ta(+course,-person,-quarter).',
    'hasposition(+person,+faculty).',
    'hasposition(+person,-faculty).',
    'hasposition(-person,+faculty).',
    'publication(+title,+person).',
    'publication(+title,-person).',
    'publication(-title,+person).',
    'inphase(+person,+prequals).',
    'inphase(+person,-prequals).',
    'inphase(-person,+prequals).',
    'courselevel(+course,+level).',
    'courselevel(+course,-level).',
    'courselevel(-course,+level).',
    'yearsinprogram(+person,+year).',
    'yearsinprogram(-person,+year).',
    'yearsinprogram(+person,-year).',
    'projectmember(+project,+person).',
    'projectmember(+project,-person).',
    'projectmember(-project,+person).']
    #'sameproject(project, project).',
    #'samecourse(course, course).',
    #'sameperson(person, person).',]
cora_target = 'samevenue'
[cora_facts, cora_pos, cora_neg] = get_cora_dataset(cora_target, folds=True)
cora_bk = ['sameauthor(+author,+author).',
          'sameauthor(+author,-author).',
          'sameauthor(-author,+author).',
          'samebib(+class,+class).',
          'samebib(+class,-class).',
          'samebib(-class,+class).',
          'sametitle(+title,+title).',
          'sametitle(+title,-title).',
          'sametitle(-title,+title).',
          'samevenue(+venue,+venue).',
          'samevenue(+venue,-venue).',
          'samevenue(-venue,+venue).',
          'author(+class,+author).',
          'author(+class,-author).',
          'author(-class,+author).',
          'title(+class,+title).',
          'title(+class,-title).',
          'title(-class,+title).',
          'venue(+class,+venue).',
          'venue(+class,-venue).',
          'venue(-class,+venue).',
          'haswordauthor(+author,+word).',
          'haswordauthor(+author,-word).',
          'haswordauthor(-author,+word).',
          'harswordtitle(+title,+word).',
          'harswordtitle(+title,-word).',
          'harswordtitle(-title,+word).',
          'haswordvenue(+venue,+word).',
          'haswordvenue(+venue,-word).',
          'haswordvenue(-venue,+word).']
nell_target = 'teamalsoknownas'
[nell_facts, nell_pos, nell_neg] = get_nell_dataset(nell_target)
nell_bk = ['athleteledsportsteam(+athlete,+sportsteam).',
          'athleteledsportsteam(+athlete,-sportsteam).',
          'athleteledsportsteam(-athlete,+sportsteam).',
          'athleteplaysforteam(+athlete,+sportsteam).',
          'athleteplaysforteam(+athlete,-sportsteam).',
          'athleteplaysforteam(-athlete,+sportsteam).',
          'athleteplaysinleague(+athlete,+sportsleague).',
          'athleteplaysinleague(+athlete,-sportsleague).',
          'athleteplaysinleague(-athlete,+sportsleague).',
          'athleteplayssport(+athlete,+sport).',
          'athleteplayssport(+athlete,-sport).',
          'athleteplayssport(-athlete,+sport).',
          'teamalsoknownas(+sportsteam,+sportsteam).',
          'teamalsoknownas(+sportsteam,-sportsteam).',
          'teamalsoknownas(-sportsteam,+sportsteam).',
          'teamplaysagainstteam(+sportsteam,+sportsteam).',
          'teamplaysagainstteam(+sportsteam,-sportsteam).',
          'teamplaysagainstteam(-sportsteam,+sportsteam).',
          'teamplaysinleague(+sportsteam,+sportsleague).',
          'teamplaysinleague(+sportsteam,-sportsleague).',
          'teamplaysinleague(-sportsteam,+sportsleague).',
          'teamplayssport(+sportsteam,+sport).',
          'teamplayssport(+sportsteam,-sport).',
          'teamplayssport(-sportsteam,+sport).']

experiments = ['imdb,uwcse', 'uwcse,imdb'] #, 'imdb,cora', 'imdb,nell', 'uwcse,nell', 'nell,imdb', 'nell,uwcse']
start = time.time()

while results['save']['experiment'] < len(experiments) and results['save']['run'] < n_runs:
    run = results['save']['run']
    experiment = experiments[results['save']['experiment']]
    if experiment not in results['results']:
        results['results'][experiment] = {'scratch': {}, 'transfer': {}}

    # imdb
    if experiment.split(',')[1] == 'imdb':
        target = imdb_target
        facts = imdb_facts[:]
        pos = imdb_pos[:]
        neg = imdb_neg[:]
        bk = imdb_bk    
    # uwcse
    elif experiment.split(',')[1] == 'uwcse':
        target = uwcse_target
        facts = uwcse_facts[:]
        pos = uwcse_pos[:]
        neg = uwcse_neg[:]
        bk = uwcse_bk
    # cora
    elif experiment.split(',')[1] == 'cora':
        target = cora_target
        facts = cora_facts[:]
        pos = cora_pos[:]
        neg = cora_neg[:]
        bk = cora_bk
    # nell
    elif experiment.split(',')[1] == 'nell':
        target = nell_target
        #facts, pos, neg = cora_facts, cora_pos, cora_neg
        facts = nell_facts[:]
        pos = split_into_folds(nell_pos, 5)
        neg = split_into_folds(nell_neg, 5)
        bk = nell_bk  

    # imdb
    if experiment.split(',')[0] == 'imdb':
        source_target = imdb_target
        source_facts = imdb_facts[:]
        source_pos = imdb_pos[:]
        source_neg = imdb_neg[:]
        source_bk = imdb_bk
    # uwcse
    elif experiment.split(',')[0] == 'uwcse':
        source_target = uwcse_target
        source_facts = uwcse_facts[:]
        source_pos = uwcse_pos[:]
        source_neg = uwcse_neg[:]
        source_bk = uwcse_bk
    # cora
    elif experiment.split(',')[0] == 'cora':
        source_target = cora_target
        source_facts = cora_facts[:]
        source_pos = cora_pos[:]
        source_neg = cora_neg[:]
        source_bk = cora_bk
    # nell
    elif experiment.split(',')[0] == 'nell':
        source_target = nell_target
        #facts, pos, neg = cora_facts, cora_pos, cora_neg
        source_facts = nell_facts[:]
        source_pos = split_into_folds(nell_pos, 5)
        source_neg = split_into_folds(nell_neg, 5)
        source_bk = nell_bk
    
    if 'fold_run' not in results['save']:
        results['save']['fold_run'] = 0

    n_folds = len(pos)
    while results['save']['fold_run'] < n_folds:
    #for i in range(n_folds):
        i = results['save']['fold_run']
        train_pos, test_pos = get_kfold_separated(i, pos)
        train_neg, test_neg = get_kfold_separated(i, neg)
        if experiment.split(',')[1] != 'nell':
            train_facts, test_facts = get_kfold_separated(i, facts)
        else:
            train_facts, test_facts = facts, facts
        
        order = [i for i in range(len(train_pos))]
        random.shuffle(order)
               
        # shuffle all examples
        random.shuffle(test_neg)
        test_neg = test_neg[:len(test_pos)]

        # Group and shuffle
        src_facts = group_folds(source_facts)
        src_pos = group_folds(source_pos)
        src_neg = group_folds(source_neg)
        random.shuffle(src_neg)
        src_neg = src_neg[:len(src_pos)]
                            
        # learning from source dataset
        background = boostsrl.modes(source_bk, [source_target], useStdLogicVariables=False, maxTreeDepth=8, nodeSize=3, numOfClauses=8)
        [model, total_revision_time, source_structured, will] = learn_model(background, boostsrl, source_target, src_pos, src_neg, src_facts, refine=None, trees=10, verbose=verbose)
        
        preds = mapping.get_preds(source_structured, source_bk)
        if verbose:
            print('\n')
            print('Predicates from source: %s \n' % preds)
            print('Source structured tree: %s \n' % source_structured)
            
        for small_train in range(len(order)):
            if small_train not in results['results'][experiment]['scratch']:
                results['results'][experiment]['scratch'][small_train] = []
            if small_train not in results['results'][experiment]['transfer']:
                results['results'][experiment]['transfer'][small_train] = []
            
            # learn from scratch in a small dataset
            s_train_pos = []
            s_train_neg = []
            if experiment.split(',')[1] != 'nell':
                s_train_facts = [] 
            else:
                s_train_facts = facts
            for k in range(small_train+1):
                s_train_pos += train_pos[order[k]]
                s_train_neg += train_neg[order[k]]
                if experiment.split(',')[1]  != 'nell':
                    s_train_facts += train_facts[order[k]]

            # shuffle all examples
            random.shuffle(s_train_neg)
            s_train_neg = s_train_neg[:len(s_train_pos)]
                                      
            # validation (10% of training set)
            validation_pos = s_train_pos[:]
            validation_neg = s_train_neg[:]
            random.shuffle(validation_pos)
            random.shuffle(validation_neg)
            validation_pos = validation_pos[:int(0.2*len(validation_pos))]
            validation_neg = validation_neg[:int(0.2*len(validation_neg))]
           
            '''# learning from small dataset
            [model, learning_time, inference_time, t_results, small_structured, will] = learn_test_model(background, boostsrl, target, s_train_pos, s_train_neg, s_train_facts, test_pos, test_neg, test_facts, trees=10, verbose=verbose)
            t_results['Learning time'] = learning_time
            t_results['Inference time'] = inference_time
            print(small_structured)
            results['results'][experiment]['small'][small_train].append(t_results)
            print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Total folds for small set: %s, Time: %s' % (experiment, run+1, i+1, 'small', small_train+1, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)'''
            
            # transfer
            mapping_rules, mapping_results = mapping.get_best(preds, bk, src_facts+src_pos, s_train_facts+s_train_pos)
            transferred_structured = transfer(source_structured, mapping_rules)
            new_target = get_transferred_target(transferred_structured)
            if verbose:
                print('\n')
                print('Best mapping found: %s \n' % mapping_rules)
                print('Tranferred structured tree: %s \n' % transferred_structured)
                print('Transferred target predicate: %s \n' % new_target)
            
            # transfer and revision theory
            background = boostsrl.modes(bk, [new_target], useStdLogicVariables=False, maxTreeDepth=8, nodeSize=3, numOfClauses=8)
            [model, total_revision_time, inference_time, t_results, structured, pl_inference_time, pl_t_results] = theory_revision(background, boostsrl, target, s_train_pos, s_train_neg, s_train_facts, validation_pos, validation_neg, test_pos, test_neg, test_facts, transferred_structured, trees=10, max_revision_iterations=10, testAfterPL=True, verbose=verbose)
            t_results['Learning time'] = total_revision_time
            t_results['Inference time'] = inference_time
            t_results['Mapping Results'] = mapping_results
            t_results['Parameter Learning Results'] = pl_t_results
            t_results['Parameter Learning Inference Time'] = pl_inference_time
            results['results'][experiment]['transfer'][small_train].append(t_results)
            print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Total folds for small set: %s, Time: %s' % (experiment, run+1, i+1, 'transfer', small_train+1, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)
            
            # learning from scratch
            [model, learning_time, inference_time, t_results, structured, will] = learn_test_model(background, boostsrl, new_target, s_train_pos, s_train_neg, s_train_facts, test_pos, test_neg, test_facts, trees=10, verbose=verbose)
            t_results['Learning time'] = learning_time
            t_results['Inference time'] = inference_time
            results['results'][experiment]['scratch'][small_train].append(t_results)
            print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Total folds for small set: %s, Time: %s' % (experiment, run+1, i+1, 'scratch', small_train+1, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)

        results['save']['fold_run'] += 1
        save(results)
    print('ueee')
    results['save']['fold_run'] = 0
    results['save']['run'] += 1
    if results['save']['run'] >= n_runs:
        results['save']['experiment'] += 1
        results['save']['run'] = 0
    save(results)
