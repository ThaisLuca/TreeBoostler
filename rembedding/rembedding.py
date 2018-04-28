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