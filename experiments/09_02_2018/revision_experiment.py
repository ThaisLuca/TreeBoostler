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
validation_size = 0.2
n_runs = 5

if os.path.isfile('revision_experiment.json'):
    with open('revision_experiment.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': {}}
    firstRun = True

def save(data):
    with open('revision_experiment.json', 'w') as fp:
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
    'ta(+course,+person,+quarter).',
    'ta(-course,+person,+quarter).',
    'ta(+course,-person,+quarter).',
    'ta(+course,+person,-quarter).',
    'ta(-course,+person,-quarter).',
    'ta(+course,-person,-quarter).',
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
cora_target = 'sameauthor'
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
nell_target = 'athleteplaysforteam'
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

experiments = ['imdb', 'uwcse', 'cora', 'nell']
start = time.time()

while results['save']['experiment'] < len(experiments) and results['save']['run'] < n_runs:
    run = results['save']['run']
    experiment = experiments[results['save']['experiment']]
    if experiment not in results['results']:
        results['results'][experiment] = {'small': {}, 'revision': {}}
    # imdb
    if experiment == 'imdb':
        target = imdb_target
        facts, pos, neg = imdb_facts, imdb_pos, imdb_neg
        bk = imdb_bk    
    # uwcse
    elif experiment == 'uwcse':
        target = uwcse_target
        facts, pos, neg = uwcse_facts, uwcse_pos, uwcse_neg
        bk = uwcse_bk
    # cora
    elif experiment == 'cora':
        target = cora_target
        facts, pos, neg = cora_facts, cora_pos, cora_neg
        bk = cora_bk
    # nell
    elif experiment == 'nell':
        target = nell_target
        #facts, pos, neg = cora_facts, cora_pos, cora_neg
        if 'nell_seed' not in results['save']:
            results['save']['nell_seed'] = random.randint(100000, 999999)
        random.seed(results['save']['nell_seed'])
        facts, pos, neg = nell_facts, split_into_folds(nell_pos, 5), split_into_folds(nell_neg, 5)
        bk = nell_bk  
    
    background = boostsrl.modes(bk, [target], useStdLogicVariables=False, maxTreeDepth=8, nodeSize=3, numOfClauses=8)
    
    if 'fold_run' not in results['save']:
        results['save']['fold_run'] = 0

    n_folds = len(pos)
    while results['save']['fold_run'] < n_folds:
    #for i in range(n_folds):
        i = results['save']['fold_run']
        train_pos, test_pos = get_kfold_separated(i, pos)
        train_neg, test_neg = get_kfold_separated(i, neg)
        if experiment != 'nell':
            train_facts, test_facts = get_kfold_separated(i, facts)
        else:
            train_facts, test_facts = facts, facts
        
        order = [i for i in range(len(train_pos))]
        random.shuffle(order)
               
        # shuffle all examples
        random.shuffle(test_neg)
        test_neg = test_neg[:len(test_pos)]
        
        for small_train in range(len(order)):
            if small_train not in results['results'][experiment]['small']:
                results['results'][experiment]['small'][small_train] = []
            if small_train not in results['results'][experiment]['revision']:
                results['results'][experiment]['revision'][small_train] = []
            
            # learn from scratch in a small dataset
            s_train_pos = []
            s_train_neg = []
            s_train_facts = [] if experiment != 'nell' else facts
            for k in range(small_train+1):
                s_train_pos += train_pos[order[k]]
                s_train_neg += train_neg[order[k]]
                if experiment != 'nell':
                    s_train_facts += train_facts[order[k]]

            # shuffle all examples
            random.shuffle(s_train_neg)
            s_train_neg = s_train_neg[:len(s_train_pos)]

            # train set used in revision and validation set
            total_train_pos, total_test_pos = get_kfold(i, pos)
            total_train_neg, total_test_neg = get_kfold(i, neg)
            if experiment != 'nell':
                total_train_facts, total_test_facts = get_kfold(i, facts)
            else:
                total_train_facts, total_test_facts = facts, facts
            r_train_pos = total_train_pos[int(validation_size*len(total_train_pos)):]
            r_train_neg = total_train_neg[int(validation_size*len(total_train_neg)):]
            r_train_facts = total_train_facts
            validation_pos = total_train_pos[:int(validation_size*len(total_train_pos))]
            validation_neg = total_train_neg[:int(validation_size*len(total_train_neg))]       
    
            # learning from small dataset
            [model, learning_time, inference_time, t_results, small_structured, will] = learn_test_model(background, boostsrl, target, s_train_pos, s_train_neg, s_train_facts, test_pos, test_neg, test_facts, trees=10, verbose=False)
            t_results['Learning time'] = learning_time
            t_results['Inference time'] = inference_time
            results['results'][experiment]['small'][small_train].append(t_results)
            print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Total folds for small set: %s, Time: %s' % (experiment, run+1, i+1, 'small', small_train+1, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)
            
            # revision theory
            [model, total_revision_time, inference_time, t_results, structured, pl_inference_time, pl_t_results] = theory_revision(background, boostsrl, target, r_train_pos, r_train_neg, r_train_facts, validation_pos, validation_neg, test_pos, test_neg, test_facts, 1.0, small_structured, trees=10, max_revision_iterations=10, verbose=False)
            t_results['Learning time'] = total_revision_time
            t_results['Inference time'] = inference_time
            results['results'][experiment]['revision'][small_train].append(t_results)
            print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Total folds for small set: %s, Time: %s' % (experiment, run+1, i+1, 'revision', small_train+1, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)
        results['save']['fold_run'] += 1
        save(results)
    results['save']['fold_run'] = 0
    results['save']['run'] += 1
    del results['save']['nell_seed']
    if results['save']['run'] >= n_runs:
        results['save']['experiment'] += 1
        results['save']['run'] = 0
    save(results)
