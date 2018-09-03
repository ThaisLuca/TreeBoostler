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
validation_size = 0.2
n_runs = 1

if os.path.isfile('transfer_experiment.json'):
    with open('transfer_experiment.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': {}}
    firstRun = True

def save(data):
    with open('transfer_experiment.json', 'w') as fp:
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

experiments = ['imdb_to_uwcse', 'imdb_to_cora', 'uwcse_to_imdb', 'uwcse_to_cora']
while results['save']['experiment'] < len(experiments) and results['save']['run'] < n_runs:
    experiment = experiments[results['save']['experiment']]
    if experiment not in results['results']:
        results['results'][experiment] = []
    # imdb to uwcse
    if experiment == 'imdb_to_uwcse':
        source_target = imdb_target
        source_facts, source_pos, source_neg = imdb_facts, imdb_pos, imdb_neg
        source_bk = imdb_bk
        target_target = uwcse_target
        target_facts, target_pos, target_neg = uwcse_facts, uwcse_pos, uwcse_neg
        target_bk = uwcse_bk
        mapping = ['workedunder(A, B) -> advisedby(A, B)',
                'director(A) -> professor(A)',
                'actor(A) -> student(A)',
                'movie(A, B) -> publication(A, B)',
                'female(A) -> student(A)'
                ]      
    # imdb to cora
    elif experiment == 'imdb_to_cora':
        source_target = imdb_target
        source_facts, source_pos, source_neg = imdb_facts, imdb_pos, imdb_neg
        source_bk = imdb_bk
        target_target = cora_target
        target_facts, target_pos, target_neg = cora_facts, cora_pos, cora_neg
        target_bk = cora_bk
        mapping = ['workedunder(A, B) -> sameauthor(A, B)',
                'movie(A, B) -> author(A, B)',
                'genre(A, B) -> haswordauthor(B, A)'
                ] 
    # uwcse to imdb
    elif experiment == 'uwcse_to_imdb':
        source_target = uwcse_target
        source_facts, source_pos, source_neg = uwcse_facts, uwcse_pos, uwcse_neg
        source_bk = uwcse_bk
        target_target = imdb_target
        target_facts, target_pos, target_neg = imdb_facts, imdb_pos, imdb_neg
        target_bk = imdb_bk
        mapping = ['advisedby(A, B) -> workedunder(A, B)',
                'professor(A) -> director(A)',
                'student(A) -> actor(A)',
                'publication(A, B) -> movie(A, B)',
                'student(A) -> female(A)'
                ]    
    # uwcse to cora
    elif experiment == 'uwcse_to_cora':
        source_target = uwcse_target
        source_facts, source_pos, source_neg = uwcse_facts, uwcse_pos, uwcse_neg
        source_bk = uwcse_bk
        target_target = cora_target
        target_facts, target_pos, target_neg = cora_facts, cora_pos, cora_neg
        target_bk = cora_bk
        mapping = ['advisedby(A, B) -> sameauthor(A, B)',
                'publication(A, B) -> author(A, B)',
                ]   
    # cora to imdb
    #elif experiment == 'cora_to_imdb':
    # cora to uwcse
    #elif experiment == 'cora_to_uwcse':    
    
    background = boostsrl.modes(source_bk, [source_target], useStdLogicVariables=False, maxTreeDepth=12, nodeSize=3, numOfClauses=12)
    
    # group source folds
    source_pos = group_folds(source_pos)
    source_neg = group_folds(source_neg)
    source_facts = group_folds(source_facts)
    
    # shuffle all examples
    random.shuffle(source_pos)
    random.shuffle(source_neg)
    source_neg = source_neg[:len(source_pos)]
      
    # learning from source
    [model, total_revision_time, structured, will] = learn_model(background, boostsrl, source_target, source_pos, source_neg, source_facts, refine=None, trees=10, verbose=False)
    
    # transfer
    transferred_structured = transfer(structured, mapping)
    
    background = boostsrl.modes(target_bk, [target_target], useStdLogicVariables=False, maxTreeDepth=12, nodeSize=3, numOfClauses=12)
    
    n_folds = len(target_pos)
    for i in range(n_folds):
        train_pos, test_pos = get_kfold(i, target_pos)
        train_neg, test_neg = get_kfold(i, target_neg)
        train_facts, test_facts = get_kfold(i, target_facts)
        
        # shuffle all examples
        random.shuffle(train_pos)
        random.shuffle(train_neg)
        train_neg = train_neg[:len(train_pos)]
        
        # separate train and validation    
        validation_pos = train_pos[:int(validation_size*len(train_pos))]
        validation_neg = train_neg[:int(validation_size*len(train_neg))]
        train_pos = train_pos[int(validation_size*len(train_pos)):]
        train_neg = train_neg[int(validation_size*len(train_neg)):]
        
        # transfer and revision
        [model, total_revision_time, inference_time, t_results, structured, pl_inference_time, pl_t_results] = theory_revision(background, boostsrl, target_target, train_pos, train_neg, train_facts, validation_pos, validation_neg, test_pos, test_neg, test_facts, 1.0, transferred_structured, trees=10, max_revision_iterations=10, verbose=False, testAfterPL=True)
        t_results['Learning time'] = total_revision_time
        t_results['Inference time'] = inference_time
        t_results['Parameter Learning time'] = pl_inference_time
        t_results['Parameter Learning results'] = pl_t_results
        results['results'][experiment].append(t_results)
    results['save']['run'] += 1
    if results['save']['run'] >= n_runs:
        results['save']['experiment'] += 1
        results['save']['run'] = 0
    save(results)