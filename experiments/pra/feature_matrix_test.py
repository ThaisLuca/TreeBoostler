#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:40:56 2018

@author: rodrigoazs
"""

import pandas as pd

data = pd.read_csv('/home/rodrigoazs/Projetos/pra/examples/results/my_nell_matrix/athleteplaysforteam/training_matrix.tsv', sep='\t', header=None)

count = 0
dict_id_to_path = {}
dict_path_to_id = {}

for index, row in data.iterrows():
    pairs = row[0]
    value = row[1]
    paths = row[2]
    path = paths.split(' -#- ')
    for i in path:
        if i not in dict_path_to_id:
            dict_id_to_path[count] = i
            dict_path_to_id[i] = count
            count += 1
            
X = []
y = []
for index, row in data.iterrows():
    pairs = row[0]
    value = row[1]
    paths = row[2]
    row = [0.0] * len(dict_path_to_id)
    path = paths.split(' -#- ')
    for i in path:
        row[dict_path_to_id[i]] = 1.0
    X.append(row)
    y.append(value)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)