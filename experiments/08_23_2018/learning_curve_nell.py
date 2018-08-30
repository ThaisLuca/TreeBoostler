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
from boostsrl import boostsrl
from sklearn.model_selection import KFold
import numpy as np
import random
import json

firstRun = False

if os.path.isfile('learning_curve_nell.json'):
    with open('learning_curve_nell.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'data': {'nell': { 'small': {}, 'revision': {} }}, 'log': [], 'folds': [], 'runs': [], 'last_run': 0}
    firstRun = True

def save(data):
    with open('learning_curve_nell.json', 'w') as fp:
        json.dump(data, fp)

dataset = 'nell'
target = 'athleteplaysforteam'
revisions = 6
smalls = 4

if firstRun:
    [facts, pos, neg] = get_nell_dataset(target)
    # shuffle all examples
    random.shuffle(pos)
    random.shuffle(neg)
    
    neg = neg[:len(pos)] # balanced   
    pos = np.array(pos)
    neg = np.array(neg)
    
    kf = KFold(5)
    fold = 0
    for train_index, test_index in kf.split(pos):
        train_pos, test_pos = pos[train_index], pos[test_index]
        train_neg, test_neg = neg[train_index], neg[test_index]
        fold += 1
        
        # shuffle all train examples
        random.shuffle(train_pos)
        random.shuffle(train_neg)
        
        results['folds'].append({'train_pos': list(train_pos), 'train_neg': list(train_neg), 'test_pos': list(test_pos), 'test_neg': list(test_neg)})
        
    for small_train_size in np.linspace(0.25, 1.0, num=smalls):
        results['data'][dataset]['small'][str(small_train_size)] = []
        results['data'][dataset]['revision'][str(small_train_size)] = {}
        for revision_threshold in np.linspace(0.5, 1.0, num=revisions):
            results['data'][dataset]['revision'][str(small_train_size)][str(revision_threshold)] = []

    for fold in range(5):
        for small_train_size in np.linspace(0.25, 1.0, num=smalls):
            results['runs'].append((fold, small_train_size))
    
    save(results)   

validation_size = 0.1
max_revision_iterations = 10

bk = ['athleteledsportsteam(+athlete,+sportsteam).',
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

background = boostsrl.modes(bk, [target], useStdLogicVariables=False, treeDepth=8, nodeSize=3, numOfClauses=8)

start = time.time()

while results['last_run'] < len(results['runs']):
    small_train_size = results['runs'][results['last_run']][1]
    fold = results['runs'][results['last_run']][0]

    train_pos = results['folds'][fold]['train_pos']
    train_neg = results['folds'][fold]['train_neg']
    test_pos = results['folds'][fold]['test_pos']
    test_neg = results['folds'][fold]['test_neg']

    # train set used in revision and validation set
    r_train_pos = train_pos[int(validation_size*len(train_pos)):]
    r_train_neg = train_neg[int(validation_size*len(train_neg)):]
    validation_pos = train_pos[:int(validation_size*len(train_pos))]
    validation_neg = train_neg[:int(validation_size*len(train_neg))]

    # learn from scratch in a small dataset
    s_train_pos = r_train_pos[:int(small_train_size*len(r_train_pos))]
    s_train_neg = r_train_neg[:int(small_train_size*len(r_train_neg))]

    # learning from small dataset
    try:
        [model, learning_time, inference_time, t_results, small_structured, will] = learn_test_model(background, boostsrl, target, s_train_pos, s_train_neg, facts, test_pos, test_neg, trees=10, verbose=False)
        t_results['Learning time'] = learning_time
        t_results['Inference time'] = inference_time
        results['data'][dataset]['small'][str(small_train_size)].append(t_results)
        save(results)
    except:
        p = 'Error in learning in fold %s and small_train_size %s' % (fold, small_train_size)
        results['log'].append(p)
        print(p)
        results['last_run'] += 1
        save(results)
        continue

    for revision_threshold in np.linspace(0.5, 1.0, num=revisions):
        # theory revision
        try:
            [model, total_revision_time, inference_time, t_results, structured] = theory_revision(background, boostsrl, target, r_train_pos, r_train_neg, facts, validation_pos, validation_neg, test_pos, test_neg, revision_threshold, small_structured.copy(), trees=10, max_revision_iterations=10, verbose=False)
            t_results['Revision time'] = total_revision_time
            t_results['Inference time'] = inference_time
            results['data'][dataset]['revision'][str(small_train_size)][str(revision_threshold)].append(t_results)
            p = 'Dataset: %s, Fold: %s, small_train_size: %s, revision_threshold: %s, time: %s' % (dataset, fold, small_train_size, revision_threshold, time.strftime('%H:%M:%S', time.gmtime(time.time()-start)))
            results['log'].append(p)
            print(p)
            save(results)
        except:
            p = 'Error in Dataset: %s, Fold: %s, small_train_size: %s, revision_threshold: %s, time: %s'% (dataset, fold, small_train_size, revision_threshold, time.strftime('%H:%M:%S', time.gmtime(time.time()-start)))
            results['log'].append(p)
            print(p)
            save(results)
            continue
        
    results['last_run'] += 1
