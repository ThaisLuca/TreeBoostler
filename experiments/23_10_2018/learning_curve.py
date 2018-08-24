'''
   Learning curve experiment
   Name:         learning_curve.py
   Author:       Rodrigo Azevedo
   Updated:      July 23, 2018
   License:      GPLv3
'''

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

#with open('domains.json', 'r') as fp:
#  data = json.load(fp)

def save(data):
    with open('learning_curve.json', 'w') as fp:
        json.dump(data, fp)
        
data = { }
for dataset in ['imdb', 'uwcse', 'nell']:
    data[dataset] = { 'small': {}, 'revision': {} }

    validation_size = 0.1
    max_revision_iterations = 10
    
    if dataset == 'imdb':
        target = 'workedunder'       
        [facts, pos, neg] = get_imdb_dataset(target)
        bk = ['workedunder(+person,+person).',
              'workedunder(+person,-person).',
              'workedunder(-person,+person).',
              'female(+person).',
              'actor(+person).',
              'director(+person).',
              'movie(+movie,+person).',
              'movie(+movie,-person).',
              'movie(-movie,+person).',
              'genre(+person,+genre).']
    elif dataset == 'uwcse':
        target = 'advisedby'       
        [facts, pos, neg] = get_uwcse_dataset(target, acceptedPredicates=[
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
        bk = ['professor(+person).',
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
    elif dataset == 'nell':
        target = 'athleteplaysforteam'
        [facts, pos, neg] = get_nell_dataset(target)
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
    
    # shuffle all examples
    random.shuffle(pos)
    random.shuffle(neg)
    
    neg = neg[:len(pos)] # balanced
    
    import numpy as np
    pos = np.array(pos)
    neg = np.array(neg)
    
    
    small_dataset_aucroc = {}
    small_dataset_aucpr = {}
    revision_dataset_aucroc = {}
    revision_dataset_aucpr = {}
    
    revisions = 6
    smalls = 4
    
    for small_train_size in np.linspace(0.25, 1.0, num=smalls):
        data[dataset]['small'][str(small_train_size)] = []
        data[dataset]['revision'][str(small_train_size)] = {}
        for revision_threshold in np.linspace(0.5, 1.0, num=revisions):
            data[dataset]['revision'][str(small_train_size)][str(revision_threshold)] = []
    
    start = time.time()
    # separate train and test
    kf = KFold(10 if dataset != 'nell' else 5)
    fold = 0
    for train_index, test_index in kf.split(pos):
        train_pos, test_pos = pos[train_index], pos[test_index]
        train_neg, test_neg = neg[train_index], neg[test_index]
        fold += 1
        
        # shuffle all train examples
        random.shuffle(train_pos)
        random.shuffle(train_neg)
            
        # train set used in revision and validation set
        r_train_pos = train_pos[int(validation_size*len(train_pos)):]
        r_train_neg = train_neg[int(validation_size*len(train_neg)):]
        validation_pos = train_pos[:int(validation_size*len(train_pos))]
        validation_neg = train_neg[:int(validation_size*len(train_neg))]
    
        # learning from complete dataset
        #[model, learning_time, inference_time, t_results, structured] = learn_test_model(background, boostsrl, target, r_train_pos, r_train_neg, facts, test_pos, test_neg, trees=1, verbose=False)
        #complete_dataset_aucroc.append(t_results['AUC ROC'])
    
        for small_train_size in np.linspace(0.25, 1.0, num=smalls): #num=4):
            # learn from scratch in a small dataset
            s_train_pos = r_train_pos[:int(small_train_size*len(r_train_pos))]
            s_train_neg = r_train_neg[:int(small_train_size*len(r_train_neg))]
            
            '''print('New small dataset')
            print('s_train_pos: '+str(len(s_train_pos)))
            print(s_train_pos)
            print('\n')'''
    
            # learning from small dataset
            try:
                [model, learning_time, inference_time, t_results, small_structured, will] = learn_test_model(background, boostsrl, target, s_train_pos, s_train_neg, facts, test_pos, test_neg, trees=10, verbose=False)
                t_results['Learning time'] = learning_time
                t_results['Inference time'] = inference_time
                data[dataset]['small'][str(small_train_size)].append(t_results)
            except:
                print('Error in learning')
                continue
    
            for revision_threshold in np.linspace(0.5, 1.0, num=revisions):
                # theory revision
                try:
                    [model, total_revision_time, inference_time, t_results, structured] = theory_revision(background, boostsrl, target, r_train_pos, r_train_neg, facts, validation_pos, validation_neg, test_pos, test_neg, revision_threshold, small_structured.copy(), trees=10, max_revision_iterations=10, verbose=False)
                    t_results['Revision time'] = total_revision_time
                    t_results['Inference time'] = inference_time
                    data[dataset]['revision'][str(small_train_size)][str(revision_threshold)].append(t_results)
                    save(data)
                    print('Dataset: %s, Fold: %s, small_train_size: %s, revision_threshold: %s, time: %s' % (dataset, fold, small_train_size, revision_threshold, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
                except:
                    print('Error in Dataset: %s, Fold: %s, small_train_size: %s, revision_threshold: %s, time: %s'% (dataset, fold, small_train_size, revision_threshold, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
                    continue
