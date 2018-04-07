"""Implementation of the Relational Embedding.


"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
import random

class REmbedding(object):
    def __init__(self):
        self.dataset = []
        self.settings = []
        self.sentences = []
        self.graph = self.Graph()

    # {'parent': ['person', 'person'] }
    def load_settings(self, st):
        self.settings = st
    
    # [('parent', ['alexis','rodrigo'])]
    def load_dataset(self, st):
        self.dataset = st
        for tupl in self.dataset:
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
        X = self.model[self.model.wv.vocab]
        return np.mean(X, axis=0)
        
    def plot_2d(self, color={}, plot_centroid=False):
        X = self.model[self.model.wv.vocab]
        print(X)
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        # create a scatter plot of the projection
        words = list(self.model.wv.vocab)
        fig = pyplot.figure(figsize=(10,10))
        fi = {}
        if plot_centroid:
            c = pca.transform(np.array(self.centroid()))
            pyplot.scatter(c[i, 0], c[i, 1], marker='x')
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
        
settings = '''parent(person,person).
male(person).
grandmother(person,person).'''
   
data = '''parent(bart,stijn).
parent(bart,pieter).
parent(luc,soetkin).
parent(willem,lieve).
parent(willem,katleen).
parent(rene,willem).
parent(rene,lucy).
parent(leon,rose).
parent(etienne,luc).
parent(etienne,an).
parent(prudent,esther).

parent(katleen,stijn).
parent(katleen,pieter).
parent(lieve,soetkin).
parent(esther,lieve).
parent(esther,katleen).
parent(yvonne,willem).
parent(yvonne,lucy).
parent(alice,rose).
parent(rose,luc).
parent(rose,an).
parent(laura,esther).

male(bart).
male(etienne).
male(leon).
male(luc).
male(pieter).
male(prudent).
male(rene).
male(stijn).
male(willem).

grandmother(esther,soetkin).
grandmother(esther,stijn).
grandmother(esther,pieter).
grandmother(yvonne,lieve).
grandmother(yvonne,katleen).
grandmother(alice,luc).
grandmother(alice,an).
grandmother(rose,soetkin).
grandmother(laura,lieve).
grandmother(laura,katleen).'''

rembedd = REmbedding()

import re
lines = settings.split('\n')
s = {}
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        s[relation] = entities

rembedd.load_settings(s)

lines = data.split('\n')
s = []
for line in lines:
    m = re.search('^(\w+)\(([\w, ]+)*\).$', line)
    if m:
        relation = m.group(1).replace(' ', '')
        entities = m.group(2).replace(' ', '').split(',')
        s.append((relation, entities))
        
rembedd.load_dataset(s)
rembedd.generate_sentences(n_sentences=10000)
rembedd.run_embedding()
rembedd.plot_2d(color={'person': 'r'}, plot_centroid=True)