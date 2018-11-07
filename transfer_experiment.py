'''
   Learning curve experiment
   Name:         transfer_experiment.py
   Author:       Rodrigo Azevedo
   Updated:      July 23, 2018
   License:      GPLv3
'''

import os
import sys
import time
#sys.path.append('../..')

from datasets.get_datasets import *
from revision import *
from transfer import *
from mapping import *
from boostsrl import boostsrl
import numpy as np
import random
import json
#import logging
#from logging import FileHandler
#from logging import Formatter

verbose=True
firstRun = False
n_runs = 40

#if not os.path.exists('log'):
#    os.makedirs('log')

#def setup_logger(name, log_file, level=logging.DEBUG):
#    formatter = logging.Formatter('%(message)s')
#    
#    handler = logging.FileHandler(log_file)        
#    handler.setFormatter(formatter)
#
#    logger = logging.getLogger(name)
#    logger.setLevel(level)
#    logger.addHandler(handler)
#
#    return logger
    
#def print(message):
#    global experiment_title
#    with open('log/' + experiment_title + '.txt', 'a') as f:
#        builtins.print(message, file=f)
#        builtins.print(message)

experiments = [
            {'source':'imdb', 'target':'uwcse', 'predicate':'workedunder', 'to_predicate':'advisedby'},
            {'source':'uwcse', 'target':'imdb', 'predicate':'advisedby', 'to_predicate':'workedunder'},
            {'source':'imdb', 'target':'cora', 'predicate':'workedunder', 'to_predicate':'samevenue'},
            {'source':'cora', 'target':'imdb', 'predicate':'samevenue', 'to_predicate':'workedunder'},
            {'source':'yeast', 'target':'twitter', 'predicate':'interaction', 'to_predicate':'follows'},
            {'source':'twitter', 'target':'yeast', 'predicate':'follows', 'to_predicate':'interaction'},
            {'source':'nell_sports', 'target':'nell_finances', 'predicate':'teamplayssport', 'to_predicate':'companyeconomicsector'},
            {'source':'nell_finances', 'target':'nell_sports', 'predicate':'companyeconomicsector', 'to_predicate':'teamplayssport'},
            #{'source':'yeast', 'target':'webkb', 'predicate':'proteinclass'},
            #{'source':'webkb', 'target':'yeast', 'predicate':'departmentof'},
            #{'source':'twitter', 'target':'webkb', 'predicate':'accounttype'},
            #{'source':'webkb', 'target':'twitter', 'predicate':'pageclass'},
            #{'source':'cora', 'target':'yago2s', 'predicate':'samevenue', 'to_predicate':'ismarriedto'},
            #{'source':'cora', 'target':'yago2s', 'predicate':'samevenue', 'to_predicate':'imports'},
            #{'source':'cora', 'target':'yago2s', 'predicate':'workedunder', 'to_predicate':'exports'},
            #{'source':'cora', 'target':'yago2s', 'predicate':'workedunder', 'to_predicate':'influences'},
            #{'source':'cora', 'target':'yago2s', 'predicate':'workedunder', 'to_predicate':'wrotemusicfor'},
            ]
            
bk = {
      'imdb': ['workedunder(+person,+person).',
              'workedunder(+person,-person).',
              'workedunder(-person,+person).',
              'female(+person).',
              'actor(+person).',
              'director(+person).',
              'movie(+movie,+person).',
              'movie(+movie,-person).',
              'movie(-movie,+person).',
              'genre(+person,+genre).'],
      'uwcse': ['professor(+person).',
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
        'projectmember(-project,+person).'],
        #'sameproject(project, project).',
        #'samecourse(course, course).',
        #'sameperson(person, person).',]
      'cora': ['sameauthor(+author,+author).',
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
              'haswordvenue(-venue,+word).'],
      'twitter': ['accounttype(+account,+type).',
                  'accounttype(+account,-type).',
                  'accounttype(-account,+type).',
                  'tweets(+account,+word).',
                  'tweets(+account,-word).',
                  'tweets(-account,+word).',
                  'follows(+account,+account).',
                  'follows(+account,-account).',
                  'follows(-account,+account).'],
      'yeast': ['location(+protein,+loc).',
                'location(+protein,-loc).',
                'location(-protein,+loc).',
                'interaction(+protein,+protein).',
                'interaction(+protein,-protein).',
                'interaction(-protein,+protein).',
                'proteinclass(+protein,+class).',
                'proteinclass(+protein,-class).',
                'proteinclass(-protein,+class).',
                'enzyme(+protein,+enz).',
                'enzyme(+protein,-enz).',
                'enzyme(-protein,+enz).',
                'function(+protein,+fun).',
                'function(+protein,-fun).',
                'function(-protein,+fun).',
                'complex(+protein,+com).',
                'complex(+protein,-com).',
                'complex(-protein,+com).',
                'phenotype(+protein,+phe).',
                'phenotype(+protein,-phe).',
                'phenotype(-protein,+phe).'],
      'nell_sports': ['athleteledsportsteam(+athlete,+sportsteam).',
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
              'teamplayssport(-sportsteam,+sport).'],
      'nell_finances': ['countryhascompanyoffice(+country,+company).',
                        'countryhascompanyoffice(+country,-company).',
                        'countryhascompanyoffice(-country,+company).',
                        'companyeconomicsector(+company,+sector).',
                        'companyeconomicsector(+company,-sector).',
                        'companyeconomicsector(-company,+sector).',
                        'economicsectorcompany(+company,+sector).',
                        'economicsectorcompany(+company,-sector).',
                        'economicsectorcompany(-company,+sector).',
                        'companyceo(+company,+person).',
                        'companyceo(+company,-person).',
                        'companyceo(-company,+person).',
                        'companyalsoknownas(+company,+company).',
                        'companyalsoknownas(+company,-company).',
                        'companyalsoknownas(-company,+company).',
                        'cityhascompanyoffice(+city,+company).',
                        'cityhascompanyoffice(+city,-company).',
                        'cityhascompanyoffice(-city,+company).',
                        'acquired(+company,+company).',
                        'acquired(+company,-company).',
                        'acquired(-company,+company).',
                        #'ceoof(+person,+company).',
                        #'ceoof(+person,-company).',
                        #'ceoof(-person,+company).',
                        'bankbankincountry(+person,+country).',
                        'bankbankincountry(+person,-country).',
                        'bankbankincountry(-person,+country).',
                        'bankboughtbank(+company,+company).',
                        'bankboughtbank(+company,-company).',
                        'bankboughtbank(-company,+company).',
                        'bankchiefexecutiveceo(+company,+person).',
                        'bankchiefexecutiveceo(+company,-person).',
                        'bankchiefexecutiveceo(-company,+person).'],              
      'yago2s': ['playsfor(+person,+team).',
    'playsfor(+person,-team).',
    'playsfor(-person,+team).',
    'hascurrency(+place,+currency).',
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
      }

if os.path.isfile('transfer_experiment.json'):
    with open('transfer_experiment.json', 'r') as fp:
        results = json.load(fp)
else:
    results = { 'results': {}, 'save': { }}
    firstRun = True

def save(data):
    with open('transfer_experiment.json', 'w') as fp:
        json.dump(data, fp)
        
if firstRun:
    results['save'] = {'experiment': 0, 'n_runs': 0, 'seed': random.randint(111111,999999) }

start = time.time()
#while results['save']['experiment'] < len(experiments):
while results['save']['n_runs'] < n_runs:
    print('Run: ' + str(results['save']['n_runs']))
    experiment = results['save']['experiment'] % len(experiments)
    try:
        #experiment = results['save']['experiment']
        experiment_title = experiments[experiment]['source'] + '->' + experiments[experiment]['target'] + '(' + experiments[experiment]['to_predicate'] + ')'
        if experiment_title not in results['results']:
            results['results'][experiment_title] = []
            
        #logger = setup_logger('logger_' + experiment_title, 'log/' + experiment_title + '.log')
        
        nbr = len(results['results'][experiment_title])
        print('Starting experiment #' + str(nbr+1) + ' for ' + experiment_title+ '\n')
    
        source = experiments[experiment]['source']
        target = experiments[experiment]['target']
        predicate = experiments[experiment]['predicate']
        to_predicate = experiments[experiment]['to_predicate']
        
        # Load source dataset
        src_total_data = datasets.load(source, bk[source], seed=results['save']['seed'])
        src_data = datasets.load(source, bk[source], target=predicate, seed=results['save']['seed'])
            
        # Group and shuffle
        src_facts = datasets.group_folds(src_data[0])
        src_pos = datasets.group_folds(src_data[1])
        src_neg = datasets.group_folds(src_data[2])
                    
        print('Start learning from source dataset\n')
                           
        # learning from source dataset
        background = boostsrl.modes(bk[source], [predicate], useStdLogicVariables=False, maxTreeDepth=3, nodeSize=2, numOfClauses=8)
        [model, total_revision_time, source_structured, will, variances] = revision.learn_model(background, boostsrl, predicate, src_pos, src_neg, src_facts, refine=None, trees=10, verbose=verbose)
        
        preds = mapping.get_preds(source_structured, bk[source])
        print('Predicates from source: %s' % preds + '\n')
            #print('Source structured tree: %s \n' % source_structured)
        
        # Load total target dataset
        tar_total_data = datasets.load(target, bk[target], seed=results['save']['seed'])
            
        n_folds = len(tar_total_data[0])
        results_save = []
        for i in range(n_folds):
            print('Starting fold ' + str(i+1) + '\n')
            
            ob_save = {}
            
            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_pos, tar_test_pos] = datasets.get_kfold_small(i, tar_total_data[0])
            else:
                t_total_data = datasets.load(target, bk[target], target=to_predicate, seed=results['save']['seed'])
                tar_train_pos = datasets.split_into_folds(t_total_data[1][0], n_folds=3, seed=results['save']['seed'])[i] + t_total_data[0][0]
            
            # transfer
            mapping_rules, mapping_results = mapping.get_best(preds, bk[target], datasets.group_folds(src_total_data[0]), tar_train_pos, forceHead=to_predicate)
            
            if verbose:
                print('Mapping Results')
                print('   Knowledge compiling time   = %s' % mapping_results['Knowledge compiling time'])
                print('   Generating paths time   = %s' % mapping_results['Generating paths time'])
                print('   Generating mappings time   = %s' % mapping_results['Generating mappings time'])
                print('   Possible mappings   = %s' % mapping_results['Possible mappings'])
                print('   Finding best mapping   = %s' % mapping_results['Finding best mapping'])
                print('   Total time   = %s' % mapping_results['Total time'])
                print('\n')
            
            transferred_structured = transfer.transfer(source_structured, mapping_rules)
            #new_target = transfer.get_transferred_target(transferred_structured)
            new_target = to_predicate
            print('Best mapping found: %s \n' % mapping_rules)
            #print('Tranferred structured tree: %s \n' % transferred_structured)
            print('Transferred target predicate: %s \n' % new_target)
            
            # Load new predicate target dataset
            tar_data = datasets.load(target, bk[target], target=new_target, seed=results['save']['seed'])
            
            # Group and shuffle
            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_facts, tar_test_facts] =  datasets.get_kfold_small(i, tar_data[0])
                [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, tar_data[1])
                [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, tar_data[2])
            else:
                [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
                to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=3, seed=results['save']['seed'])
                to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=3, seed=results['save']['seed'])
                [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
                [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)
    
            # transfer and revision theory
            background = boostsrl.modes(bk[target], [new_target], useStdLogicVariables=False, maxTreeDepth=3, nodeSize=2, numOfClauses=8)
            [model, t_results, structured, pl_t_results] = revision.theory_revision(background, boostsrl, target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, transferred_structured, trees=10, max_revision_iterations=10, verbose=verbose)
            t_results['Mapping results'] = mapping_results
            t_results['Parameter Learning results'] = pl_t_results
            ob_save['transfer'] = t_results
            print('Dataset: %s, Fold: %s, Type: %s, Time: %s' % (experiment_title, i+1, 'transfer', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)
            print('\n')
            
            print('Start learning from scratch in target domain\n')
            
            # learning from scratch
            [model, t_results, structured, will, variances] = revision.learn_test_model(background, boostsrl, new_target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, trees=10, verbose=verbose)
            ob_save['scratch'] = t_results
            print('Dataset: %s, Fold: %s, Type: %s, Time: %s' % (experiment_title, i+1, 'scratch', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print(t_results)
            print('\n')
            
            results_save.append(ob_save)
        results['results'][experiment_title].append(results_save)
    except Exception as e:
        print(e)
        print('Error in experiment of ' + experiment_title)
        pass
    results['save']['experiment'] += 1
    results['save']['n_runs'] += 1
    save(results)
