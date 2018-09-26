"""Implementation of the Relational Embedding.


"""

from gensim.models import Word2Vec
from gensim import matutils
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
import random

def get_statistics(source_sentences):
    source_statistics = {}
    for key, value in source_sentences.items():
        a = key.split()
        if len(a) == 3:
            if a[1] not in source_statistics:
                source_statistics[a[1]] = []
            source_statistics[a[1]].append((key, value))
    #for key, value in source_statistics.items():
    #    sorted(source_statistics[key], key=lambda x: x[1], reverse=True)
    return source_statistics

def get_set(statistics):
    s = set()
    for key, value in statistics.items():
        for v in value:
            s.add(v[0])
    return s

def count_matches(source_mapped, target):
    return len(source_mapped.intersection(target))

def union_sets(source_mapped, target):
    return len(source_mapped.union(target))

def invert(predicate):
    if predicate[0] == '_':
        return predicate[1:]
    else:
        return '_' + predicate

def map_set(mapping, source):
    mapped = set()
    for value in source:
        predicates = value.split()
        new_predicates = []
        for predicate in predicates:
            if predicate in mapping:
                new_predicates.append(mapping[predicate])
            elif predicate[1:] in mapping:
                new_predicates.append(invert(mapping[predicate[1:]]))
            else:
                break
        if len(new_predicates) < 3:
            continue
        mapped.add(' '.join(new_predicates))
    return mapped            
            
class REmbedding(object):
    def __init__(self):
        self.dataset = []
        self.settings = []
        self.sentences = []
        self.types = set()
        self.predicates = set()
        self.graph = self.Graph()

    # {'parent': ['person', 'person'] }
    def load_settings(self, st):
        for s in st:
            for i in st[s]:
                self.types.add(i)
        self.settings = st
    
    # [('parent', ['alexis','rodrigo'])]
    def load_dataset(self, st):
        self.dataset = st
        for tupl in self.dataset:
            self.predicates.add(tupl[0])
            type1 = self.settings[tupl[0]][0]
            type2 = self.settings[tupl[0]][1] if len(tupl[1]) > 1 else self.settings[tupl[0]][0]
            sub = type1 + '_' + tupl[1][0]
            obj = type2 + '_' + tupl[1][1] if len(tupl[1]) > 1 else type2 + '_' + tupl[1][0]
            self.graph.add_relation(sub, tupl[0], obj, True if len(tupl[1]) > 1 else False)
            
    def generate_sentences(self, max_depth=4, n_sentences=100000):
        import time
        start_time = time.time()
        self.sentences = []
        for i in range(n_sentences):
            node = self.graph.nodes[random.choice(list(self.graph.nodes))]
            clauses = {}
            sentence = [] #[str(node)]
            i_depth = 1
            while(i_depth < max_depth):
                if node not in clauses:
                    clauses[node] = set()
                edg = node.edges.difference(clauses[node])
                if len(edg) == 0:
                    break
                edge = random.choice(list(edg))
                if edge[1] not in clauses:
                    clauses[edge[1]] = set()
                clauses[node].add((edge[0], edge[1]))
                clauses[edge[1]].add((edge[0][1:] if edge[0][:1] == '_' else '_' + edge[0], node))
                sentence.append(str(edge[0]))
                #sentence.append(str(edge[1]))
                node = edge[1]
                i_depth += 1
            self.sentences.append(sentence)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    def run_embedding(self, **kwargs):
        self.model = Word2Vec(self.sentences, **kwargs)
        
    def centroid(self):
        return np.mean(self.model[self.model.wv.vocab], axis=0)
    
    def type_centroid(self):
        typ = {}
        for word in list(self.model.wv.vocab):
            s = word.split('_')
            if len(s) > 1 and len(s[0]) > 0:
                if s[0] not in typ:
                    typ[s[0]] = []
                typ[s[0]].append(self.model[word])
        for t in typ:
            typ[t] = np.mean(typ[t], axis=0)
        return typ
    
    def most_similar_predicate(self, vector):
        #self.model.wv.similarity()
        top = self.model.wv.similar_by_vector(vector, topn=len(self.model.wv.vocab))
        real = []
        for t in top:
            s = t[0].split('_')
            if len(s) <= 1 or len(s[0]) == 0:
                real.append(t)
        return real
    
    def most_similar_type(self, vector):
        types = self.type_centroid()
        distances = []
        for t in types:
            distances.append((t, np.dot(matutils.unitvec(vector), matutils.unitvec(types[t]))))
        distances = sorted(distances, key=lambda x: x[1], reverse=True)
        return distances
        
    def plot_2d(self, color={}, plot_centroid=False):
        X = self.model[self.model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        words = list(self.model.wv.vocab)
        pyplot.figure(figsize=(10,10))
        if plot_centroid:
            c = pca.transform(np.array([self.centroid()]))
            pyplot.scatter(c[0, 0], c[0, 1], marker='x')
            centroids = self.type_centroid()
            for cen in centroids:
                c = pca.transform(np.array([centroids[cen]]))
                pyplot.scatter(c[0, 0], c[0, 1], marker='x', c=color[cen])
                pyplot.annotate(cen, xy=(c[0, 0], c[0, 1]))
        fi = {}
        for i, word in enumerate(words):
            spl = word.split('_')
            if len(spl) == 1 or len(spl[0]) == 0:
                pyplot.scatter(result[i, 0], result[i, 1])
                pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
            else:
                key = spl[0]
                if key not in fi:
                    pyplot.scatter(result[i, 0], result[i, 1], c=color[key], label=key)
                    fi[key] = 1
                else:
                    pyplot.scatter(result[i, 0], result[i, 1], c=color[key])
                #pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.legend()
        pyplot.show()
        
    def plot_2d_vectors(self, vectors):
        pca = PCA(n_components=2)
        pca.fit(self.model[self.model.wv.vocab])
        X = [value for key, value in vectors.items()]
        result = pca.transform(X)
        words = list(vectors)
        pyplot.figure(figsize=(10,10))
        for i, word in enumerate(words):
            pyplot.scatter(result[i, 0], result[i, 1])
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.legend()
        pyplot.show()

    class Graph(object):
        def __init__(self):
            self.nodes = {}
    
        def add_relation(self, subject, relation, object_, symmetry=True):
            if subject not in self.nodes:
                self.nodes[subject] = self.Node(subject)
            if object_ not in self.nodes:
                self.nodes[object_] = self.Node(object_)
            self.nodes[subject].add_edge(relation, self.nodes[object_], symmetry)
            
        class Node(object):
            def __init__(self, name):
                self.name = name
                self.edges = set()
            
            def add_edge(self, relation, node, symmetry=True):
                self._add_edge(relation, node)
                if symmetry:
                    node._add_edge('_'+relation, self)
            
            def _add_edge(self, relation, node):
                self.edges.add((relation, node))
        
            def __str__(self):
                return str(self.name)
                
            def __hash__(self):
                return hash(self.name)
                
            def __eq__(self, other):
                return str(self) == str(other)


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
publication(title,person).
inphase(person,prequals).
courselevel(course,level).
yearsinprogram(person,year).
projectmember(project,person).
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

source.generate_sentences(max_depth=4, n_sentences=1000000)
target.generate_sentences(max_depth=4, n_sentences=1000000)

source_sentences = [' '.join(i) for i in source.sentences]
target_sentences = [' '.join(i) for i in target.sentences]

import collections

source_sentences = dict(collections.Counter(source_sentences))
target_sentences = dict(collections.Counter(target_sentences))

source_statistics = get_statistics(source_sentences)
target_statistics = get_statistics(target_sentences)

source_set = get_set(source_statistics)
target_set = get_set(target_statistics)

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': '_advisedby', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'female': 'student', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'female': 'director', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'female': 'director', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': '_advisedby', 'female': 'director', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': '_advisedby', 'movie': '_publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': '_advisedby', 'movie': '_publication', 'actor': 'professor', 'director': 'student'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': '_advisedby', 'movie': '_publication', 'actor': 'professor', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': '_publication', 'actor': 'professor', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': '_publication', 'actor': 'professor', 'director': 'student'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'student', 'director': 'professor', 'genre': 'taughtby'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'student', 'director': 'professor', 'genre': 'inphase'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'professor', 'director': 'professor', 'genre': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'professor', 'director': 'professor', 'genre': '_inphase'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'professor', 'director': 'professor', 'genre': 'courselevel'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'professor', 'director': 'professor', 'genre': 'yearsinprogram'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'advisedby', 'movie': 'publication', 'actor': 'professor', 'director': 'professor', 'genre': '_projectmember'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

mapping = {'workedunder': 'tempadvisedby', 'movie': 'publication', 'actor': 'student', 'director': 'professor'}
mapped_set = map_set(mapping, source_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, target_set),  union_sets(mapped_set, target_set), (count_matches(mapped_set, target_set)/union_sets(mapped_set, target_set))))

########################

mapping = {'advisedby': 'workedunder', 'publication': 'movie', 'student': 'actor', 'professor': 'director'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))

mapping = {'advisedby': '_workedunder', 'publication': 'movie', 'student': 'actor', 'professor': 'director'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))

mapping = {'advisedby': 'workedunder', 'publication': '_movie', 'student': 'actor', 'professor': 'director'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))

mapping = {'advisedby': 'workedunder', 'publication': '_movie'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))

mapping = {'advisedby': 'workedunder', 'publication': 'movie', 'student': 'actor', 'professor': 'director', 'inphase': 'genre'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))

mapping = {'advisedby': 'workedunder', 'publication': 'movie', 'student': 'actor', 'professor': 'director', 'inphase': 'tempadvisedby'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))

mapping = {'advisedby': 'workedunder', 'publication': 'movie', 'student': 'director', 'professor': 'actor', 'inphase': 'tempadvisedby'}
mapped_set = map_set(mapping, target_set)
print(mapping)
print('Jaccard: %s/%s = %s' % (count_matches(mapped_set, source_set),  union_sets(mapped_set, source_set), (count_matches(mapped_set, source_set)/union_sets(mapped_set, source_set))))



'''source.run_embedding()
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

#target.most_similar_type(source_type_centroid['person']+transformation)
#target.most_similar_type(source_type_centroid['movie']+transformation)
#target.most_similar_type(source_type_centroid['genre']+transformation)

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
source.most_similar_predicate(target.model['student'] + transformation)'''