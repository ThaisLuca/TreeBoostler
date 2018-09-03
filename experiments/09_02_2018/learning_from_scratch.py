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

firstRun = False
n_runs = 1

if os.path.isfile('learning_from_scratch.json'):
    with open('learning_from_scratch.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': {}}
    firstRun = True

def save(data):
    with open('learning_from_scratch.json', 'w') as fp:
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

experiments = ['imdb', 'uwcse', 'cora']
start = time.time()

while results['save']['experiment'] < len(experiments) and results['save']['run'] < n_runs:
    experiment = experiments[results['save']['experiment']]
    if experiment not in results['results']:
        results['results'][experiment] = []
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
    
    background = boostsrl.modes(bk, [target], useStdLogicVariables=False, maxTreeDepth=12, nodeSize=3, numOfClauses=12)
    
    # group source folds
    #pos = group_folds(pos)
    #neg = group_folds(neg)
    #facts = group_folds(facts)
      
    n_folds = len(pos)
    for i in range(n_folds):
        train_pos, test_pos = get_kfold(i, pos)
        train_neg, test_neg = get_kfold(i, neg)
        train_facts, test_facts = get_kfold(i, facts)
        
        # shuffle all examples
        #random.shuffle(train_pos)
        random.shuffle(train_neg)
        random.shuffle(test_neg)
        train_neg = train_neg[:len(train_pos)]
        test_neg = test_neg[:len(test_pos)]
        
        # transfer and revision
        [model, learning_time, inference_time, t_results, structured, will] = learn_test_model(background, boostsrl, target, train_pos, train_neg, train_facts, test_pos, test_neg, test_facts, refine=None, trees=10, verbose=False)
        t_results['Learning time'] = learning_time
        t_results['Inference time'] = inference_time
        results['results'][experiment].append(t_results)
        print('Dataset: %s, Fold: %s, Time: %s' % (experiment, i+1, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
        print(t_results)
    results['save']['run'] += 1
    if results['save']['run'] >= n_runs:
        results['save']['experiment'] += 1
        results['save']['run'] = 0
    save(results)