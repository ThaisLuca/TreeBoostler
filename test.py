'''
   Testing algorithm
   Name:         test.py
   Author:       Rodrigo Azevedo
   Updated:      July 22, 2018
   License:      GPLv3
'''

from datasets.get_datasets import *
from revision import *
from boostsrl import boostsrl

target = 'playsfor'
[facts, pos, neg] = get_yago2s_dataset(target, acceptedPredicates=[
'hascurrency',
'hascapital',
'hasacademicadvisor',
'haswonprize',
'participatedin',
'owns',
'isinterestedin',
'livesin',
'happenedin',
'holdspoliticalposition',
'diedin',
'actedin',
'iscitizenof',
'worksat',
'directed',
'dealswith',
'wasbornin',
'created',
'isleaderof',
'haschild',
'ismarriedto',
'imports',
'hasmusicalrole',
'influences',
'isaffiliatedto',
'isknownfor',
'ispoliticianof',
'graduatedfrom',
'exports',
'edited',
'wrotemusicfor'])

bk = ['playsfor(+person,+team).',
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

background = boostsrl.modes(bk, [target], useStdLogicVariables=False, treeDepth=8, nodeSize=3, numOfClauses=8)

train_pos = pos
train_neg = neg
train_facts = facts

model = boostsrl.train(background, train_pos, train_neg, train_facts) #, refine=[';cancer(A) :- smokes(A).;false;true', 'false;cancer(A) :- friends(A,B), smokes(B).;false;false'])

time = model.traintime()

will = model.get_will_produced_tree()

structured = model.get_structured_tree()

results = boostsrl.test(model, pos[:100], neg[:100], facts)

results.summarize_results()

'''target = 'workedunder'
[facts, pos, neg] = get_imdb_dataset(target)

bk = boostsrl.example_data('background')
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

background = boostsrl.modes(bk, [target], useStdLogicVariables=False, treeDepth=8, nodeSize=3, numOfClauses=8)

train_pos = pos
train_neg = neg
train_facts = facts

model = boostsrl.train(background, train_pos, train_neg, train_facts) #, refine=[';cancer(A) :- smokes(A).;false;true', 'false;cancer(A) :- friends(A,B), smokes(B).;false;false'])

time = model.traintime()

will = model.get_will_produced_tree()

structured = model.get_structured_tree()

#candidates = get_cantidates(structured, 12)

results = boostsrl.test(model, pos[:100], neg[:100], facts)

results.summarize_results()'''