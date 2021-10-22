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
from tboostsrl import tboostsrl
import numpy as np
import random
import json
import pickle

#verbose=True
source_balanced = False
balanced = False
firstRun = False
folds = 3

nodeSize = 2
numOfClauses = 8
maxTreeDepth = 3
trees = 10

if not os.path.exists('transfer-experiments'):
    os.makedirs('transfer-experiments')

def print_function(message):
    global experiment_title
    global nbr
    if not os.path.exists('transfer-experiments/' + experiment_title):
        os.makedirs('transfer-experiments/' + experiment_title)
    with open('transfer-experiments/' + experiment_title + '/' + str(nbr) + '_' + experiment_title + '.txt', 'a') as f:
        print(message, file=f)
        print(message)

def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def get_next_node(node, next):
  """
      Add next nodes of tree to list of rules

      Args:
          node(list): current branch to be added to list
          next(list): next branch of tree given the current node
      Returns:
          all rules from a node
  """

  if not node:
    return next
  b = node.split(',')
  b.append(next)
  return ','.join(b)


def get_rules(structure, treenumber=1):
  """
      Sweep through a branch of the tree 
      to get all rules

      Args:
          structure(list): tree struct
          treenumber(int): number of the tree to be processed
      Returns:
          all rules learned in the given branch
  """

  target = structure[0]
  nodes = structure[1]
  tree = treenumber-1

  rules = []
  for path, value in nodes.items():
    node = target + ' :- ' + value + '.' if not path else value + '.'
    true =  'true'  if get_next_node(path, 'true')  in nodes else 'false'
    false = 'true'  if get_next_node(path, 'false') in nodes else 'false'
    rules.append(';'.join([str(tree), path, node, true, false]))
  return rules

def write_to_file(data, filename, op='w'):
  """
      Write data to a specific file

      Args:
          data(list): information to be written
          filename(str): name of file in which the data will be written
          op(str): 'w' to create a new file or 'a' to append data to a new file if exists
  """
  with open(filename, op) as f:
      for line in data:
          f.write(line + '\n')

def save_experiment(data):
    if not os.path.exists('transfer-experiments/' + experiment_title):
        os.makedirs('transfer-experiments/' + experiment_title)
    results = []
    if os.path.isfile('transfer-experiments/' + experiment_title + '/' + experiment_title + '.json'):
        with open('transfer-experiments/' + experiment_title + '/' + experiment_title + '.json', 'r') as fp:
            results = json.load(fp)
    results.append(data)
    with open('transfer-experiments/' + experiment_title + '/' + experiment_title + '.json', 'w') as fp:
        json.dump(results, fp)

def get_number_experiment():
    results = []
    if os.path.isfile('transfer-experiments/' + experiment_title + '/' + experiment_title + '.json'):
        with open('transfer-experiments/' + experiment_title + '/' + experiment_title + '.json', 'r') as fp:
            results = json.load(fp)
    return len(results)

def save(data):
    with open('transfer-experiments/transfer_experiment.json', 'w') as fp:
        json.dump(data, fp)

def save_pickle_file(nodes, _id, source, target, filename):
    with open(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, filename), 'wb') as file:
        pickle.dump(nodes, file)

def match_bk_source(sources):
  """
      Match nodes to source background

      Args:
          nodes(dict): dictionary with nodes in order of depth
          sources(list): all predicates found in source background
      Returns:
          all nodes learned by the model
  """
  source_match = {}
  for source in sources:
    if(source.split('(')[0] not in source_match):
      source_match[source.split('(')[0]] = source.replace('.', '').replace('+', '').replace('-', '')
  return source_match

def get_all_rules_from_tree(structures):
  """
      Sweep through the relational tree 
      to get all relational rules

      Args:
          structure(list): tree struct
      Returns:
          all rules learned by the model
  """

  rules = []
  for i in range(len(structures)):
    rules += get_rules(structures[i], treenumber=i+1)
  return rules

def deep_first_search_nodes(structure, matches={}, trees=[]):
  """
      Uses Deep-First Search to return all nodes

      Args:
          structure(list/dict/str/float): something to be added to the list
          trees: list to hold tree nodes. As we are using recursion, its default is empty.
      Returns:
          all nodes learned by the model
  """
  if(isinstance(structure, list)):
    for element in structure:
      trees = deep_first_search_nodes(element, matches, trees)
    return trees
  elif(isinstance(structure, dict)):
    node_number = 0
    nodes = {}
    for key in structure:
      if(isinstance(structure[key], str)):
        nodes[node_number] = matches.get(structure[key].split('(')[0], structure[key])
        node_number += 1
    if(nodes):
      trees.append(nodes)
    return trees
  else:
    return trees

experiments = [
            {'id': '1', 'source':'imdb', 'target':'uwcse', 'predicate':'workedunder', 'to_predicate':'advisedby', 'arity': 2},
            #{'id': '2', 'source':'uwcse', 'target':'imdb', 'predicate':'advisedby', 'to_predicate':'workedunder', 'arity': 2},
            {'id': '3', 'source':'imdb', 'target':'cora', 'predicate':'workedunder', 'to_predicate':'samevenue', 'arity': 2},
            {'id': '4', 'source':'cora', 'target':'imdb', 'predicate':'samevenue', 'to_predicate':'workedunder', 'arity': 2},
            {'id': '5', 'source':'cora', 'target':'imdb', 'predicate':'sametitle', 'to_predicate':'workedunder', 'arity': 2},
            {'id': '6', 'source':'imdb', 'target':'cora', 'predicate':'workedunder', 'to_predicate':'sametitle', 'arity': 2},
            ##{'id': '5', 'source':'uwcse', 'target':'cora', 'predicate':'advisedby', 'to_predicate':'samevenue', 'arity': 2},
            ##{'id': '6', 'source':'cora', 'target':'uwcse', 'predicate':'samevenue', 'to_predicate':'advisedby', 'arity': 2},
            {'id': '7', 'source':'yeast', 'target':'twitter', 'predicate':'proteinclass', 'to_predicate':'accounttype', 'arity': 2},
            {'id': '8', 'source':'twitter', 'target':'yeast', 'predicate':'accounttype', 'to_predicate':'proteinclass', 'arity': 2},
            {'id': '9', 'source':'nell_sports', 'target':'nell_finances', 'predicate':'teamplayssport', 'to_predicate':'companyeconomicsector', 'arity': 2},
            #{'id': '10', 'source':'nell_finances', 'target':'nell_sports', 'predicate':'companyeconomicsector', 'to_predicate':'teamplayssport', 'arity': 2},
            #{'id': '11', 'source':'uwcse', 'target':'webkb', 'predicate':'advisedby', 'to_predicate':'departmentof', 'arity':2},
            #{'id': '12', 'source':'webkb', 'target':'yeast', 'predicate':'departmentof', 'to_predicate':'proteinclass', 'arity':2},
            #{'id': '13', 'source': 'yago2s', 'target': 'yeast', 'predicate': 'wasbornin', 'to_predicate': 'proteinclass', 'arity': 2},
            #{'id': '14', 'source': 'yeast', 'target': 'yago2s', 'predicate': 'proteinclass', 'to_predicate': 'wasbornin', 'arity': 2},
            #{'id': '15', 'source': 'yeast', 'target': 'yeast2', 'predicate': 'proteinclass', 'to_predicate': 'gene', 'arity': 2},
            #{'id': '16', 'source': 'yeast', 'target': 'fly', 'predicate': 'proteinclass', 'to_predicate': 'gene', 'arity': 2},
            #{'id': '48', 'source':'twitter', 'target':'facebook', 'predicate':'follows', 'to_predicate':'edge', 'arity': 2},
            #{'id': '49', 'source':'imdb', 'target':'facebook', 'predicate':'workedunder', 'to_predicate':'edge','arity': 2},
            #{'id': '50', 'source':'uwcse', 'target':'facebook', 'predicate':'advisedby', 'to_predicate':'edge', 'arity': 2},
            ]

bk = {
      'imdb': ['workedunder(+person,+person).',
              'workedunder(+person,-person).',
              'workedunder(-person,+person).',
              #'recursion_workedunder(+person,`person).',
              #'recursion_workedunder(`person,+person).',
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
        #'recursion_advisedby(`person,+person).',
        #'recursion_advisedby(+person,`person).',
        'tempadvisedby(+person,+person).',
        'tempadvisedby(+person,-person).',
        'tempadvisedby(-person,+person).',
        'ta(+course,+person,+quarter).',
        'ta(-course,-person,+quarter).',
        'ta(+course,-person,-quarter).',
        'ta(-course,+person,-quarter).',
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
        'projectmember(-project,+person).',
        'sameproject(+project,+project).',
        'sameproject(+project,-project).',
        'sameproject(-project,+project).',
        'samecourse(+course,+course).',
        'samecourse(+course,-course).',
        'samecourse(-course,+course).',
        'sameperson(+person,+person).',
        'sameperson(+person,-person).',
        'sameperson(-person,+person).',],
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
              #'recursion_samevenue(+venue,`venue).',
              #'recursion_samevenue(`venue,+venue).',
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
              'haswordtitle(+title,+word).',
              'haswordtitle(+title,-word).',
              'haswordtitle(-title,+word).',
              'haswordvenue(+venue,+word).',
              'haswordvenue(+venue,-word).',
              'haswordvenue(-venue,+word).'],
      'webkb': ['coursepage(+page).',
                'facultypage(+page).',
                'studentpage(+page).',
                'researchprojectpage(+page).',
                'linkto(+id,+page,+page).',
                'linkto(+id,-page,-page).',
                'linkto(-id,-page,+page).',
                'linkto(-id,+page,-page).',
                'has(+word,+page).',
                'has(+word,-page).',
                'has(-word,+page).',
                'hasalphanumericword(+id).',
                'allwordscapitalized(+id).',
                'instructorsof(+page,+page).',
                'instructorsof(+page,-page).',
                'instructorsof(-page,+page).',
                'hasanchor(+word,+page).',
                'hasanchor(+word,-page).',
                'hasanchor(-word,+page).',
                'membersofproject(+page,+page).',
                'membersofproject(+page,-page).',
                'membersofproject(-page,+page).',
                'departmentof(+page,+page).',
                'departmentof(+page,-page).',
                'departmentof(-page,+page).',
                'pageclass(+page,+class).',
                'pageclass(+page,-class).',
                'pageclass(-page,+class).'],
      'twitter': ['accounttype(+account,+type).',
                  'accounttype(+account,-type).',
                  'accounttype(-account,+type).',
                  #'typeaccount(+type,`account).',
                  #'typeaccount(`type,+account).',
                  'tweets(+account,+word).',
                  'tweets(+account,-word).',
                  'tweets(-account,+word).',
                  'follows(+account,+account).',
                  'follows(+account,-account).',
                  'follows(-account,+account).',
                  'recursion_accounttype(+account,`type).',
                  'recursion_accounttype(`account,+type).',],
      'yeast': ['location(+protein,+loc).',
                'location(+protein,-loc).',
                'location(-protein,+loc).',
                'interaction(+protein,+protein).',
                'interaction(+protein,-protein).',
                'interaction(-protein,+protein).',
                'proteinclass(+protein,+class).',
                'proteinclass(+protein,-class).',
                'proteinclass(-protein,+class).',
                #'classprotein(+class,`protein).',
                #'classprotein(`class,+protein).',
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
                'phenotype(-protein,+phe).',
                'recursion_proteinclass(+protein,`class).',
                'recursion_proteinclass(`protein,+class).'],
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
              'teamplayssport(-sportsteam,+sport).',
              'teamplayssport(+sportsteam,`sport).',
              'teamplayssport(`sportsteam,+sport).'],
              #'recursion_teamplayssport(`sportsteam,+sport).',
              #'recursion_teamplayssport(+sportsteam,`sport).'],
      'nell_finances': ['countryhascompanyoffice(+country,+company).',
                        'countryhascompanyoffice(+country,-company).',
                        'countryhascompanyoffice(-country,+company).',
                        'companyeconomicsector(+company,+sector).',
                        'companyeconomicsector(+company,-sector).',
                        'companyeconomicsector(-company,+sector).',
                        'economicsectorcompany(+sector,`company).',
                        'economicsectorcompany(`sector,+company).',
                        #'recursion_economicsectorcompany(+sector,`company).',
                        #'recursion_economicsectorcompany(`sector,+company).',
                        #'economicsectorcompany(+sector,+company).',
                        #'economicsectorcompany(+sector,-company).',
                        #'economicsectorcompany(-sector,+company).',
                        #'ceoeconomicsector(+person,+sector).',
                        #'ceoeconomicsector(+person,-sector).',
                        #'ceoeconomicsector(-person,+sector).',
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
    'wrotemusicfor(-person,+media).'],
    'facebook': ['edge(+person,+person).',
            'edge(+person,-person).',
            'edge(-person,+person).',
            'middlename(+person,+middlename).',
            'middlename(+person,-middlename).',
            'middlename(-person,+middlename).',
            'lastname(+person,+lastname).',
            'lastname(+person,-lastname).',
            'lastname(-person,+lastname).',
            'educationtype(+person,+educationtype).',
            'educationtype(+person,-educationtype).',
            'educationtype(-person,+educationtype).',
            'workprojects(+person,+workprojects).',
            'workprojects(+person,-workprojects).',
            'workprojects(-person,+workprojects).',
            'educationyear(+person,+educationyear).',
            'educationyear(+person,-educationyear).',
            'educationyear(-person,+educationyear).',
            'educationwith(+person,+educationwith).',
            'educationwith(+person,-educationwith).',
            'educationwith(-person,+educationwith).',
            'location(+person,+location).',
            'location(+person,-location).',
            'location(-person,+location).',
            'workwith(+person,+workwith).',
            'workwith(+person,-workwith).',
            'workwith(-person,+workwith).',
            'workenddate(+person,+workenddate).',
            'workenddate(+person,-workenddate).',
            'workenddate(-person,+workenddate).',
            'languages(+person,+languages).',
            'languages(+person,-languages).',
            'languages(-person,+languages).',
            'religion(+person,+religion).',
            'religion(+person,-religion).',
            'religion(-person,+religion).',
            'political(+person,+political).',
            'political(+person,-political).',
            'political(-person,+political).',
            'workemployer(+person,+workemployer).',
            'workemployer(+person,-workemployer).',
            'workemployer(-person,+workemployer).',
            'hometown(+person,+hometown).',
            'hometown(+person,-hometown).',
            'hometown(-person,+hometown).',
            'educationconcentration(+person,+educationconcentration).',
            'educationconcentration(+person,-educationconcentration).',
            'educationconcentration(-person,+educationconcentration).',
            'workfrom(+person,+workfrom).',
            'workfrom(+person,-workfrom).',
            'workfrom(-person,+workfrom).',
            'workstartdate(+person,+workstartdate).',
            'workstartdate(+person,-workstartdate).',
            'workstartdate(-person,+workstartdate).',
            'worklocation(+person,+worklocation).',
            'worklocation(+person,-worklocation).',
            'worklocation(-person,+worklocation).',
            'educationclasses(+person,+educationclasses).',
            'educationclasses(+person,-educationclasses).',
            'educationclasses(-person,+educationclasses).',
            'workposition(+person,+workposition).',
            'workposition(+person,-workposition).',
            'workposition(-person,+workposition).',
            'firstname(+person,+firstname).',
            'firstname(+person,-firstname).',
            'firstname(-person,+firstname).',
            'birthday(+person,+birthday).',
            'birthday(+person,-birthday).',
            'birthday(-person,+birthday).',
            'educationschool(+person,+educationschool).',
            'educationschool(+person,-educationschool).',
            'educationschool(-person,+educationschool).',
            'name(+person,+name).',
            'name(+person,-name).',
            'name(-person,+name).',
            'gender(+person,+gender).',
            'gender(+person,-gender).',
            'gender(-person,+gender).',
            'educationdegree(+person,+educationdegree).',
            'educationdegree(+person,-educationdegree).',
            'educationdegree(-person,+educationdegree).',
            'locale(+person,+locale).',
            'locale(+person,-locale).',
            'locale(-person,+locale).'],
    'yeast2': ['cites(+paper,+paper)',
               'cites(+paper,-paper)',
               'cites(-paper,+paper)',
               'gene(+paper,+gene)',
               'gene(+paper,-gene)',
               'gene(-paper,+gene)',
               'journal(+paper,+journal)',
               'journal(+paper,-journal)',
               'journal(-paper,+journal)',
               'author(+paper,+author))',
               'author(+paper,-author)',
               'author(-paper,+author)',
               'chem(+paper,+chemical)',
               'chem(+paper,-chemical)',
               'chem(-paper,+chemical)',
               'aff(+paper,+institute)',
               'aff(+paper,-institute)',
               'aff(-paper,+institute)',
               'aspect(+paper,+gene,+R)',
               'aspect(-paper,-gene,+R)',
               'aspect(+paper,-gene,-R)',
               'aspect(-paper,+gene,-R)'],
    'fly': ['journal(+paper,+journal)',
            'journal(+paper,-journal)',
            'journal(-paper,+journal)',
            'author(+paper,+author))',
            'author(+paper,-author)',
            'author(-paper,+author)',
            'cites(+paper,+paper)',
            'cites(+paper,-paper)',
            'cites(-paper,+paper)',
            'cites(+paper,+paper)',
            'cites(+paper,-paper)',
            'cites(-paper,+paper)',
            'gene(+paper,+gene)',
            'gene(+paper,-gene)',
            'gene(-paper,+gene)',
            'aspect(+paper,+gene,+R)',
            'aspect(-paper,-gene,+R)',
            'aspect(+paper,-gene,-R)',
            'aspect(-paper,+gene,-R)',
            'gp(+gene,+protein)',
            'gp(+gene,-protein)',
            'gp(-gene,+protein)',
            'genetic(+gene,+gene)',
            'genetic(+gene,-gene)',
            'genetic(-gene,+gene)'
            'physical(+gene,+gene)',
            'physical(+gene,-gene)',
            'physical(-gene,+gene)'],
      }
    
#if os.path.isfile('transfer_experiment.json'):
#    with open('transfer_experiment.json', 'r') as fp:
#        results = json.load(fp)
#else:
#    results = { 'save': { }}
#    firstRun = True
    
results = {}
for experiment in experiments:
    
    target = experiment['target']
            
    results = {'save': { }}
    firstRun = True
    
    results['save'] = {
        'experiment': 0,
        'n_runs': 0,
        'seed': 441773,
        'source_balanced' : False,
        'balanced' : False,
        'folds' : n_runs,
        'nodeSize' : 2,
        'numOfClauses' : 8,
        'maxTreeDepth' : 3
        }

    if 'nodes' in locals():
            nodes.clear()

    if 'structured' in locals():
        structured.clear()
        
    # Load total target dataset
    tar_total_data = datasets.load(target, bk[target], seed=results['save']['seed'])
    
    if target in ['nell_sports', 'nell_finances', 'yago2s']:
        n_runs = params.N_FOLDS
    else:
        n_runs = len(tar_total_data[0])

    _id = experiment['id']
    source = experiment['source']
    target = experiment['target']
    predicate = experiment['predicate']
    to_predicate = experiment['to_predicate']
    
    experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
    
    print_function('Starting experiment #' + str(nbr) + ' for ' + experiment_title+ '\n')
    nbr = get_number_experiment() + 1
    
    start = time.time()

    while results['save']['n_runs'] < n_runs:
        print('Run: ' + str(results['save']['n_runs'] + 1))
        
        # Load source dataset
        src_total_data = datasets.load(source, bk[source], seed=results['save']['seed'])
        src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=results['save']['seed'])

        # Group and shuffle
        src_facts = datasets.group_folds(src_data[0])
        src_pos = datasets.group_folds(src_data[1])
        src_neg = datasets.group_folds(src_data[2])

        print_function('Start learning from source dataset\n')

        print_function('Source train facts examples: %s' % len(src_facts))
        print_function('Source train pos examples: %s' % len(src_pos))
        print_function('Source train neg examples: %s\n' % len(src_neg))

        # learning from source dataset
        background = tboostsrl.modes(bk[source], [predicate], useStdLogicVariables=False, maxTreeDepth=maxTreeDepth, nodeSize=nodeSize, numOfClauses=numOfClauses)
        [model, total_revision_time, source_structured, will, variances] = revision.learn_model(background, tboostsrl, predicate, src_pos, src_neg, src_facts, refine=None, trees=trees, print_function=print_function)

        #if not os.path.exists('resources/' + experiment_title):
        #    os.makedirs('resources/' + experiment_title)

        # Get the list of predicates from source tree          
        #nodes = deep_first_search_nodes(source_structured, match_bk_source(set(bk[source])))
        #save_pickle_file(nodes, _id, source, target, 'source_tree_nodes.pkl')
        #save_pickle_file(source_structured, _id, source, target, 'source_structured_nodes.pkl')

        #refine_structure = get_all_rules_from_tree(source_structured)
        #write_to_file(refine_structure, os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, 'refine.txt'))

        #source_structured = load_pickle_file(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, 'source_structured_nodes.pkl'))

        # Load total target dataset
        #tar_total_data = datasets.load(target, bk[target], seed=results['save']['seed'])

        if target in ['nell_sports', 'nell_finances', 'yago2s']:
            n_folds = folds
        else:
            n_folds = len(tar_total_data[0])

        results_save = []
        
        for i in range(n_folds):
            print_function('Starting fold ' + str(i+1) + '\n')

            ob_save = {}
            
            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_pos, tar_test_pos] = datasets.get_kfold_small(i, tar_total_data[0])
            else:
                t_total_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=results['save']['seed'])
                tar_train_pos = datasets.split_into_folds(t_total_data[1][0], n_folds=n_folds, seed=results['save']['seed'])[i] + t_total_data[0][0]

            # Load new predicate target dataset
            tar_data = datasets.load(target, bk[target], target=to_predicate, balanced=balanced, seed=results['save']['seed'])

            # Group and shuffle
            if target not in ['nell_sports', 'nell_finances', 'yago2s']:
                [tar_train_facts, tar_test_facts] =  datasets.get_kfold_small(i, tar_data[0])
                [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, tar_data[1])
                [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, tar_data[2])
            else:
                [tar_train_facts, tar_test_facts] =  [tar_data[0][0], tar_data[0][0]]
                to_folds_pos = datasets.split_into_folds(tar_data[1][0], n_folds=n_folds, seed=results['save']['seed'])
                to_folds_neg = datasets.split_into_folds(tar_data[2][0], n_folds=n_folds, seed=results['save']['seed'])
                [tar_train_pos, tar_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
                [tar_train_neg, tar_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)
            
            random.shuffle(tar_train_pos)
            random.shuffle(tar_train_neg)

            print_function('Target train facts examples: %s' % len(tar_train_facts))
            print_function('Target train pos examples: %s' % len(tar_train_pos))
            print_function('Target train neg examples: %s\n' % len(tar_train_neg))
            
            print_function('Target test facts examples: %s' % len(tar_test_facts))
            print_function('Target test pos	 examples: %s' % len(tar_test_pos))
            print_function('Target test neg examples: %s\n' % len(tar_test_neg))

            # generate transfer file
            transferred_structured = source_structured
            tr_file = transfer.get_transfer_file(bk[source], bk[target], predicate, to_predicate, searchArgPermutation=True, allowSameTargetMap=False)
            new_target = to_predicate

            # transfer and revision theory
            background = tboostsrl.modes(bk[target], [to_predicate], useStdLogicVariables=False, maxTreeDepth=maxTreeDepth, nodeSize=nodeSize, numOfClauses=numOfClauses)
            [model, t_results, structured, pl_t_results] = revision.theory_revision(background, tboostsrl, target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, transferred_structured, transfer=tr_file, trees=trees, max_revision_iterations=1, print_function=print_function)

            t_results['parameter'] = pl_t_results
            ob_save['transfer'] = t_results
            print_function('Dataset: %s, Fold: %s, Type: %s, Time: %s' % (experiment_title, i+1, 'Transfer (trRDN-B)', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print_function(t_results)
            print_function('\n')

            # learning from scratch (RDN-B)
            [model, t_results, structured, will, variances] = revision.learn_test_model(background, tboostsrl, target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, trees=trees, print_function=print_function)
            ob_save['rdn_b_' + str(amount)] = t_results
            print_function('Dataset: %s, Fold: %s, Type: %s, Time: %s' % (experiment_title, i+1, 'Scratch (RDN-B)', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            print_function(t_results)
            print_function('\n')

            # learning from scratch (RDN)
            #background = tboostsrl.modes(bk[target], [new_target], useStdLogicVariables=False, maxTreeDepth=3, nodeSize=2, numOfClauses=20)
            #[model, t_results, structured, will, variances] = revision.learn_test_model(background, tboostsrl, new_target, tar_train_pos, tar_train_neg, tar_train_facts, tar_test_pos, tar_test_neg, tar_test_facts, trees=1, print_function=print_function)
            #ob_save['rdn_' + str(amount)] = t_results
            #print_function('Dataset: %s, Fold: %s, Type: %s, Time: %s' % (experiment_title, i+1, 'Scratch (RDN)', time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
            #print_function(t_results)
            #print_function('\n')

            results_save.append(ob_save)
        save_experiment(results_save)

        results['save']['experiment'] += 1
        results['save']['n_runs'] += 1
        save(results)
