"""Implementation of the Relational Embedding.


"""

from gensim.models import Word2Vec
from gensim import matutils
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
import random

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
            
    def generate_sentences(self, max_depth=10, n_sentences=1000000):
        import time
        start_time = time.time()
        self.sentences = []
        for i in range(n_sentences):
            node = self.graph.nodes[random.choice(list(self.graph.nodes))]
            clauses = {}
            sentence = [str(node)]
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
                sentence.append(str(edge[1]))
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

"""Implementation of Transfer process


"""

def PerformTransfer(source_data, source_settings, target_data, target_settings, n_runs=5, max_depth=10, n_sentences=1000000):
    def average_ranking_cosine(ranks):
        avg = {}
        n = len(ranks)
        for rank in ranks:
            for entity in rank:
                if entitie[0] not in avg:
                    avg[entity[0]] = entity[1]/n
                else:
                    avg[entity[0]] += entity[1]/n
        avg = avg.items()
        avg = sorted(avg, key=lambda x: x[1], reverse=True)
        return avg
    
    def average_ranking(ranks):
        avg = {}
        n = len(ranks)
        for rank in ranks:
            for i in range(len(rank)):
                pos = len(rank)-i
                entity = rank[i]
                if entity[0] not in avg:
                    avg[entity[0]] = [pos/n, entity[1]/n]
                else:
                    avg[entity[0]][0] += pos/n
                    avg[entity[0]][1] += entity[1]/n
        avg = avg.items()
        avg = sorted(avg, key=lambda x: x[1][0], reverse=True)
        return avg
        
    def is_consistent(source, target):
        # check arity
        s = source
        t = target if target[0] != '_' else target[1:]
        inverse = False if target[0] != '_' else True
        if len(ss[s]) != len(ts[t]):
            return False
        if len(ss[s]) == 1:
            if mapping_types[ss[s][0]] == ts[t][0]:
                return True
            else:
                return False
        else:
            if inverse == True:
                if mapping_types[ss[s][0]] == ts[t][1] and mapping_types[ss[s][1]] == ts[t][0]:
                    return True
                else:
                    return False
            else:
                if mapping_types[ss[s][0]] == ts[t][0] and mapping_types[ss[s][1]] == ts[t][1]:
                    return True
                else:
                    return False
        return True
    
    source = REmbedding()
    target = REmbedding()
    
    source.load_settings(source_settings)
    source.load_dataset(source_data)
    
    target.load_settings(target_settings)
    target.load_dataset(target_data)
    
    results_types = {}
    results_predicates = {}
    
    for i in range(n_runs):
        source.generate_sentences(max_depth=max_depth, n_sentences=n_sentences)
        target.generate_sentences(max_depth=max_depth, n_sentences=n_sentences)
        source.run_embedding()
        target.run_embedding()
        #source.plot_2d(color={'person': 'r', 'movie': 'b', 'genre':'g'}, plot_centroid=True)
        #plt.savefig('source' + int(i))
        #target.plot_2d(color={'person': 'r', 'faculty': 'b', 'course': 'g', 'title': 'y'}, plot_centroid=True)
        #plt.savefig('target' + int(i))
        
        
        source_centroid = source.centroid()
        target_centroid = target.centroid()
        
        source_type_centroid = source.type_centroid()
        transformation = target_centroid - source_centroid
        
        for typ in source.types:
            similars = target.most_similar_type(source_type_centroid[typ]+transformation)
            if typ not in results_types:
                results_types[typ] = [similars]
            else:
                results_types[typ].append(similars)
                
        for pred in source.predicates:
            similars = target.most_similar_predicate(source.model[pred]+transformation)
            if pred not in results_predicates:
                results_predicates[pred] = [similars]
            else:
                results_predicates[pred].append(similars)
    
    rtypes = results_types.copy()
    ptypes = results_predicates.copy()
    for key, value in results_types.items():
        results_types[key] = average_ranking(value)
    for key, value in results_predicates.items():
        results_predicates[key] = average_ranking(value)
                
    #return [results_types, results_predicates, (rtypes, ptypes)]

    # search for better types pair
    mapping_types = {}
    source_mappeds = set()
    target_mappeds = set()
    map_rank = []
    for key, value in results_types.items():
        for vec in range(len(value)):
            map_rank.append(([key, value[vec][0]], value[vec][1]))
    map_rank = sorted(map_rank, key=lambda x: (x[1][0], x[1][1]), reverse=True)
    
    for i in map_rank:
        #print(i)
        if i[0][0] not in source_mappeds and i[0][1] not in target_mappeds:
            source_mappeds.add(i[0][0])
            target_mappeds.add(i[0][1])
            mapping_types[i[0][0]] = i[0][1]
    
    # search for better predicates pair
    mapping_predicates = {}
    source_mappeds = set()
    target_mappeds = set()
    map_rank = []
    for key, value in results_predicates.items():
        for vec in range(len(value)):
            print(key + str(value[vec]))
            map_rank.append(([key, value[vec][0]], value[vec][1]))
    map_rank = sorted(map_rank, key=lambda x: (x[1][0], x[1][1]), reverse=True)
    
    for i in map_rank:
        #print(i)
        if (i[0][0] not in source_mappeds and '_'+i[0][0] not in source_mappeds) and (i[0][1] not in target_mappeds and '_'+i[0][1] not in target_mappeds) and is_consistent(i[0][0], i[0][1]) == True:
            source_mappeds.add(i[0][0])
            target_mappeds.add(i[0][1])
            mapping_predicates[i[0][0]] = i[0][1]

    return {'types':mapping_types, 'predicates': mapping_predicates}

# Running code
import re               
source_settings = '''workedunder(person,person).
female(person).
movie(movie,person).
genre(person,genre).
actor(person).
director(person).
'''

lines = source_settings.split('\n')
ss = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        ss[relation] = entities

sd = []
with open('test/imdb.pl') as f:
    for line in f:
        m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
        if m:
            relation = m.group(1).replace(' ', '')
            entities = m.group(2).replace(' ', '').split(',')
            sd.append((relation, entities))
            
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
ts = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        ts[relation] = entities

td = []
with open('test/uwcselearn.pl') as f:
    for line in f:
        m = re.search('^(\w+)\(ai,([\w, ]+)*\).$', line)
        if m:
            relation = m.group(1).replace(' ', '')
            entities = m.group(2).replace(' ', '').split(',')
            if relation in ts:
                td.append((relation, entities))
                
t = PerformTransfer(sd,ss,td,ts, 1, max_depth=10)