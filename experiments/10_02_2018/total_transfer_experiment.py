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



experiments = [
            {'source':'imdb', 'target':'uwcse', 'predicate':'workedunder'},
            {'source':'imdb', 'target':'uwcse', 'predicate':'movie'},
            {'source':'uwcse', 'target':'imdb', 'predicate':'advisedby'},
            {'source':'uwcse', 'target':'imdb', 'predicate':'publication'},
            {'source':'imdb', 'target':'cora', 'predicate':'workedunder'},
            {'source':'imdb', 'target':'cora', 'predicate':'movie'},
            {'source':'uwcse', 'target':'cora', 'predicate':'advisedby'},
            {'source':'uwcse', 'target':'cora', 'predicate':'publication'},
            {'source':'cora', 'target':'imdb', 'predicate':'samevenue'},
            {'source':'cora', 'target':'uwcse', 'predicate':'samevenue'},
            {'source':'imdb', 'target':'nell', 'predicate':'workedunder'},
            {'source':'uwcse', 'target':'nell', 'predicate':'advisedby'},
            #{'source':'nell', 'target':'yago', 'predicate':'athleteplaysforteam'},
            ]

imdb_data = datasets.get_imdb_dataset(None, folds=True)
uwcse_data = datasets.get_uwcse_dataset(None, folds=True, acceptedPredicates=['professor','student','advisedby','tempadvisedby','hasposition','publication','inphase','courselevel','yearsinprogram','projectmember'])
cora_data = datasets.get_cora_dataset(None, folds=True)

def get_data(data, predicate, seed=None):
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
    
    #nell_data = datasets.get_nell_dataset(None, seed=results['save']['seed'])
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
    
    yago_bk = ['hascurrency(+place,+currency).',
    'hascurrency(+place,-currency).',
    'hascurrency(-place,+currency).',
    'hascapital(+place,+place).',
    'hascapital(+place,-place).',
    'hascapital(-place,+place).',
    'hasacademicadvisor(+person,+person).',
    'hasacademicadvisor(+person,-person).',
    'hasacademicadvisor(-person,+person).',
    'haswonprize(+person,+prize).',
    'haswonprize(+person,-prize).',
    'haswonprize(-person,+prize).',
    'participatedin(+place,+event).',
    'participatedin(+place,-event).',
    'participatedin(-place,+event).',
    'owns(+institution,+institution).',
    'owns(+institution,-institution).',
    'owns(-institution,+institution).',
    'isinterestedin(+person,+concept).',
    'isinterestedin(+person,-concept).',
    'isinterestedin(-person,+concept).',
    'livesin(+person,+place).',
    'livesin(+person,-place).',
    'livesin(-person,+place).',
    'happenedin(+event,+place).',
    'happenedin(+event,-place).',
    'happenedin(-event,+place).',
    'holdspoliticalposition(+person,+politicalposition).',
    'holdspoliticalposition(+person,-politicalposition).',
    'holdspoliticalposition(-person,+politicalposition).',
    'diedin(+person,+place).',
    'diedin(+person,-place).',
    'diedin(-person,+place).',
    'actedin(+person,+media).',
    'actedin(+person,-media).',
    'actedin(-person,+media).',
    'iscitizenof(+person,+place).',
    'iscitizenof(+person,-place).',
    'iscitizenof(-person,+place).',
    'worksat(+person,+institution).',
    'worksat(+person,-institution).',
    'worksat(-person,+institution).',
    'directed(+person,+media).',
    'directed(+person,-media).',
    'directed(-person,+media).',
    'dealswith(+place,+place).',
    'dealswith(+place,-place).',
    'dealswith(-place,+place).',
    'wasbornin(+person,+place).',
    'wasbornin(+person,-place).',
    'wasbornin(-person,+place).',
    'created(+person,+media).',
    'created(+person,-media).',
    'created(-person,+media).',
    'isleaderof(+person,+place).',
    'isleaderof(+person,-place).',
    'isleaderof(-person,+place).',
    'haschild(+person,+person).',
    'haschild(+person,-person).',
    'haschild(-person,+person).',
    'ismarriedto(+person,+person).',
    'ismarriedto(+person,-person).',
    'ismarriedto(-person,+person).',
    'imports(+person,+material).',
    'imports(+person,-material).',
    'imports(-person,+material).',
    'hasmusicalrole(+person,+musicalrole).',
    'hasmusicalrole(+person,-musicalrole).',
    'hasmusicalrole(-person,+musicalrole).',
    'influences(+person,+person).',
    'influences(+person,-person).',
    'influences(-person,+person).',
    'isaffiliatedto(+person,+team).',
    'isaffiliatedto(+person,-team).',
    'isaffiliatedto(-person,+team).',
    'isknownfor(+person,+theory).',
    'isknownfor(+person,-theory).',
    'isknownfor(-person,+theory).',
    'ispoliticianof(+person,+place).',
    'ispoliticianof(+person,-place).',
    'ispoliticianof(-person,+place).',
    'graduatedfrom(+person,+institution).',
    'graduatedfrom(+person,-institution).',
    'graduatedfrom(-person,+institution).',
    'exports(+place,+material).',
    'exports(+place,-material).',
    'exports(-place,+material).',
    'edited(+person,+media).',
    'edited(+person,-media).',
    'edited(-person,+media).',
    'wrotemusicfor(+person,+media).',
    'wrotemusicfor(+person,-media).',
    'wrotemusicfor(-person,+media).']
    
    # Load source dataset
    # imdb
    if data == 'imdb':
        if predicate:
            [facts, pos, neg] = datasets.target_examples(predicate, imdb_data)
        else:
            facts = imdb_data[0]
            pos = imdb_data[1]
            neg = imdb_data[2]
        bk = imdb_bk
    # uwcse
    elif data == 'uwcse':
        if predicate:
            [facts, pos, neg] = datasets.target_examples(predicate, uwcse_data)
        else:
            facts = uwcse_data[0]
            pos = uwcse_data[1]
            neg = uwcse_data[2]
        bk = uwcse_bk
    # cora
    elif data == 'cora':
        if predicate:
            [facts, pos, neg] = datasets.target_examples(predicate, cora_data)
        else:
            facts = cora_data[0]
            pos = cora_data[1]
            neg = cora_data[2]
        bk = cora_bk
    # nell
    elif data == 'nell':
        [facts, pos, neg] = datasets.get_nell_dataset(predicate, seed=seed)
        random.seed(seed)
        random.shuffle(pos)
        random.shuffle(neg)
        pos = datasets.split_into_folds(pos, 5)
        neg = datasets.split_into_folds(neg, 5)
        bk = nell_bk
    # yago
    elif data == 'yago':
        accepted = ['hascurrency','hascapital','hasacademicadvisor','participatedin','haswonprize','participatedin','owns','isinterestedin','livesin','happenedin','holdspoliticalposition','diedin','actedin','iscitizenof','worksat','directed','dealswith','wasbornin','created','isleaderof','haschild','ismarriedto','imports','hasmusicalrole','influences','isaffiliatedto','isknownfor','ispoliticianof','graduatedfrom','exports','edited','wrotemusicfor']
        [facts, pos, neg] = datasets.get_yago2s_dataset(predicate, acceptedPredicates=accepted, seed=seed)
        random.seed(seed)
        random.shuffle(pos)
        random.shuffle(neg)
        pos = datasets.split_into_folds(pos, 5)
        neg = datasets.split_into_folds(neg, 5)
        bk = yago_bk
        
    return [bk, facts, pos, neg]

firstRun = False
n_runs = 1
verbose = True

if os.path.isfile('total_transfer_experiment.json'):
    with open('total_transfer_experiment.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': { 'fold_run': 0 }}
    firstRun = True

def save(data):
    with open('total_transfer_experiment.json', 'w') as fp:
        json.dump(data, fp)
        
if firstRun:
    results['save'] = {'experiment': 0, 'run': 0, 'fold_run': 0, 'seed': random.randint(111111,999999) }

start = time.time()
while results['save']['experiment'] < len(experiments) and results['save']['run'] < n_runs:
    run = results['save']['run']
    experiment = results['save']['experiment']
    if experiment not in results['results']:
        results['results'][experiment] = {'scratch': {}, 'transfer': {}}

    source = experiments[experiment]['source']
    target = experiments[experiment]['target']
    predicate = experiments[experiment]['predicate']
    
    # Load source dataset
    [src_bk, src_facts, src_pos, src_neg] = get_data(source, predicate, seed=results['save']['seed'])
        
    # Group and shuffle
    random.seed(results['save']['seed'])
    if source not in ['nell', 'yago']:
        src_facts = datasets.group_folds(src_facts)
    src_pos = datasets.group_folds(src_pos)
    src_neg = datasets.group_folds(src_neg)
    random.shuffle(src_neg)
    src_neg = src_neg[:len(src_pos)]
                       
    # learning from source dataset
    background = boostsrl.modes(src_bk, [predicate], useStdLogicVariables=False, maxTreeDepth=8, nodeSize=3, numOfClauses=8)
    [model, total_revision_time, source_structured, will] = learn_model(background, boostsrl, predicate, src_pos, src_neg, src_facts, refine=None, trees=10, verbose=verbose)
    
    preds = mapping.get_preds(source_structured, src_bk)
    if verbose:
        print('\n')
        print('Predicates from source: %s \n' % preds)
        print('Source structured tree: %s \n' % source_structured)
    
    # Load total target dataset
    [tar_bk, tar_facts, tar_pos, tar_neg] = get_data(target, None, seed=results['save']['seed'])
        
    n_folds = len(tar_pos)
    while results['save']['fold_run'] < n_folds:
        
        i = results['save']['fold_run']
        if i not in results['results'][experiment]['scratch']:
            results['results'][experiment]['scratch'][i] = []
        if i not in results['results'][experiment]['transfer']:
            results['results'][experiment]['transfer'][i] = []
                
        [tar_train_pos, tar_test_pos] = datasets.get_kfold(i, tar_pos)
        
        # transfer
        mapping_rules, mapping_results = mapping.get_best(preds, tar_bk, src_facts+src_pos, tar_train_pos)
        transferred_structured = transfer(source_structured, mapping_rules)
        new_target = get_transferred_target(transferred_structured)
        if verbose:
            print('\n')
            print('Best mapping found: %s \n' % mapping_rules)
            print('Tranferred structured tree: %s \n' % transferred_structured)
            print('Transferred target predicate: %s \n' % new_target)
        
        # Load new predicate target dataset
        [tar_bk, tar_facts, tar_pos, tar_neg] = get_data(target, new_target, seed=results['save']['seed'])
        
        # Group and shuffle
        random.seed(results['save']['seed'])
        if target not in ['nell', 'yago']:
            [tar_train_facts, tar_test_facts] = datasets.get_kfold(i, tar_facts)
        else:
            tar_train_facts = tar_facts
            tar_test_facts = tar_facts
        [tar_train_pos, tar_test_pos] = datasets.get_kfold(i, tar_pos)
        [tar_train_neg, tar_test_neg] = datasets.get_kfold(i, tar_neg)
        random.shuffle(tar_train_neg)
        random.shuffle(tar_test_neg)
        tar_train_neg = tar_train_neg[:len(tar_train_pos)]
        tar_test_neg = tar_test_neg[:len(tar_test_pos)]
        
        # validation (10% of training set)
        random.seed(results['save']['seed'])
        validation_pos = tar_train_pos[:]
        validation_neg = tar_train_neg[:]
        random.shuffle(validation_pos)
        random.shuffle(validation_neg)
        validation_pos = validation_pos[:int(0.1*len(validation_pos))]
        validation_neg = validation_neg[:int(0.1*len(validation_neg))]
        
        # transfer and revision theory
        background = boostsrl.modes(tar_bk, [new_target], useStdLogicVariables=False, maxTreeDepth=8, nodeSize=3, numOfClauses=8)
        [model, total_revision_time, inference_time, t_results, structured, pl_inference_time, pl_t_results] = theory_revision(background, boostsrl, target, tar_train_pos, tar_train_neg, tar_train_facts, validation_pos, validation_neg, tar_test_pos, tar_test_neg, tar_test_facts, transferred_structured, trees=10, max_revision_iterations=10, testAfterPL=True, verbose=verbose)
        t_results['Learning time'] = total_revision_time
        t_results['Inference time'] = inference_time
        t_results['Mapping Results'] = mapping_results
        t_results['Parameter Learning Results'] = pl_t_results
        t_results['Parameter Learning Inference Time'] = pl_inference_time
        results['results'][experiment]['transfer'][i].append(t_results)
        print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Time: %s' % (experiment, run+1, i+1, 'transfer', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
        print(t_results)
        
        # learning from scratch
        [model, learning_time, inference_time, t_results, structured, will] = learn_test_model(background, boostsrl, new_target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, trees=10, verbose=verbose)
        t_results['Learning time'] = learning_time
        t_results['Inference time'] = inference_time
        results['results'][experiment]['scratch'][i].append(t_results)
        print('Dataset: %s, Run: %s, Fold: %s, Type: %s, Time: %s' % (experiment, run+1, i+1, 'scratch', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
        print(t_results)
        
        results['save']['fold_run'] += 1
        save(results)

    results['save']['fold_run'] = 0
    results['save']['run'] += 1
    if results['save']['run'] >= n_runs:
        results['save']['experiment'] += 1
        results['save']['run'] = 0
    save(results)
