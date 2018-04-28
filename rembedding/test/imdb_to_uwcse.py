#import os, sys
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from rembedding.rembedding import REmbedding
import re

source_settings = '''workedunder(person,person).
female(person).
movie(movie,person).
genre(person,genre).
actor(person).
director(person).
'''

source = REmbedding()
target = REmbedding()

lines = source_settings.split('\n')
s = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        s[relation] = entities
        
source.load_settings(s)

s = []
with open('imdb.pl') as f:
    for line in f:
        m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
        if m:
            relation = m.group(1).replace(' ', '')
            entities = m.group(2).replace(' ', '').split(',')
            s.append((relation, entities))
            
source.load_dataset(s)

target_settings = '''professor(person).
student(person).
hasposition(person,faculty).
taughtby(course,person).
advisedby(person,person).
tempadvisedby(person,person).
ta(course,person).
publication(title,person).
'''

lines = target_settings.split('\n')
sett = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        sett[relation] = entities
        
target.load_settings(sett)

s = []
with open('uwcselearn.pl') as f:
    for line in f:
        m = re.search('^(\w+)\(ai,([\w, ]+)*\).$', line)
        if m:
            relation = m.group(1).replace(' ', '')
            entities = m.group(2).replace(' ', '').split(',')
            if relation in sett:
                s.append((relation, entities))
            
target.load_dataset(s)

source.generate_sentences(n_sentences=10000000)
target.generate_sentences(n_sentences=10000000)
source.run_embedding()
target.run_embedding()
source.plot_2d(color={'person': 'r', 'movie': 'b', 'genre':'g'}, plot_centroid=True)
target.plot_2d(color={'person': 'r', 'faculty': 'b', 'course': 'g', 'title': 'y'}, plot_centroid=True)

source_centroid = source.centroid()
target_centroid = target.centroid()

source_type_centroid = source.type_centroid()
transformation = target_centroid - source_centroid

#######################################################
## plot das constelações
#source_words = {}
#for word in source.model.wv.vocab:
#    s = word.split('_')
#    if len(s) == 1 or len(s[0]) == 0:
#        source_words['s_' + word] = source.model[word]
#        
#for word in source_type_centroid:
#    source_words['s_type_' + word] = source_type_centroid[word]
#    
#source.plot_2d_vectors(source_words)
# 
#target_words = {}       
#for word in target.model.wv.vocab:
#    s = word.split('_')
#    if len(s) == 1 or len(s[0]) == 0:
#        target_words['t_' + word] = target.model[word]
# 
#target_type_centroid = target.type_centroid()
#for word in target_type_centroid:
#    target_words['t_type_' + word] = target_type_centroid[word]
#    
#target.plot_2d_vectors(target_words)
#
## plot das constelações juntas
#merge_words = source_words.copy()
#merge_words.update(target_words)
#for word in merge_words:
#    # aplica transformacao
#    if word[:2] == 's_':
#        merge_words[word] = merge_words[word] + transformation
#target.plot_2d_vectors(merge_words)
#######################################################

target.most_similar_type(source_type_centroid['person']+transformation)
target.most_similar_type(source_type_centroid['movie']+transformation)
target.most_similar_type(source_type_centroid['genre']+transformation)

target.most_similar_predicate(source.model['workedunder']+transformation)
target.most_similar_predicate(source.model['movie']+transformation)
target.most_similar_predicate(source.model['actor']+transformation)
target.most_similar_predicate(source.model['director']+transformation)
target.most_similar_predicate(source.model['genre']+transformation)
target.most_similar_predicate(source.model['female']+transformation)


transformation = source_centroid - target_centroid 
source.most_similar_predicate(target.model['publication'] + transformation)
source.most_similar_predicate(target.model['taughtby'] + transformation)
source.most_similar_predicate(target.model['professor'] + transformation)
source.most_similar_predicate(target.model['student'] + transformation)