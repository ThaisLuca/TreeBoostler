'''
   Functions to handle transfer learning of boosted trees
   Name:         transfer.py
   Author:       Rodrigo Azevedo
   Updated:      August 31, 2018
   License:      GPLv3
'''
from boostedrevision import *
import copy
import re

'''transfer_map = ['workedunder(A, B) -> advisedby(B, A)',
            'director(A) -> professor(A)',
            'actor(A) -> student(A)',
            'movie(A, B) -> publication(B, B)'
            ]

structured = [['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)',
   'false': 'teste(A,B)',
   'false,false': 'movie(B,A)'},
  {'false,true': [6.83e-08, 77, 0], 'false,false,false': [6.83e-08, 77, 0], 'false,false,true': [6.83e-08, 77, 0], 'true': [0.0, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [0.0, 77, 0], 'true': [2.23e-07, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [5.67e-08, 77, 0], 'true': [3.26e-07, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [5.27e-08, 77, 0], 'true': [0.0, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [0.0, 77, 0], 'true': [0.0, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [1.83e-08, 77, 0], 'true': [0.0, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [3.57e-08, 77, 0], 'true': [0.0, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [0.0, 77, 0], 'true': [2.98e-08, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [2.89e-08, 77, 0], 'true': [7.88e-08, 0, 77]}],
 ['workedunder(A, B)',
  {'': 'director(B), movie(C, A), movie(C, B)'},
  {'false': [0.0, 77, 0], 'true': [5.37e-08, 0, 77]}]]'''

def literal_to_str(literal):
    '''Generate literal string from tuple'''
    return literal[0] + '(' + literal[1] + ')'
    
def transfer_variables(variables, from_map, to_map):
    '''Transfer variables of a clause given a mapping'''
    read = {}
    ret = []
    values = variables.split(',')
    f = from_map.split(',')
    t = to_map.split(',')
    for i in range(len(values)):
        read[f[i].strip()] = values[i].strip()
    for i in range(len(t)):
        ret.append(read[t[i].strip()])
    return ', '.join(ret)
    
def get_mapping_struct(mapping):
    '''Generate mapping structure'''
    ret = {}
    for m in mapping:
        match = re.findall('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', m)
        if match:
            ret[match[0][0]] = (match[1][0], match[0][1], match[1][1])
    return ret
    
def transfer_literal(literal, mapping_struct):
    if literal[0] in mapping_struct:
        m = mapping_struct[literal[0]]
        return (m[0], transfer_variables(literal[1], m[1], m[2]))
    else:
        return None
    
def transfer(structured, mapping):
    '''Transfer structure according to mapping'''
    copied = copy.deepcopy(structured)
    mapping_struct = get_mapping_struct(mapping)
    for struct in copied:
        match = re.match('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', struct[0])
        if match:
            transfered = transfer_literal(match.groups(), mapping_struct)
            if transfered:
                struct[0] = literal_to_str(transfered)
            else:
                raise(Exception('Attempted to transfer head to a None mapping.'))
        else:
             raise(Exception('Attempted to transfer head that does not exist.'))
        # nodes with no literals should be removed among with its subtree
        remove_nodes = []
        new_nodes = {}
        for key, value in struct[1].items():
            if ','.join((key.split(','))[:-1]) not in remove_nodes:
                match = re.findall('([a-zA-Z_0-9]*)\s*\(([a-zA-Z_0-9,\s]*)\)', value)
                new_clause = []
                if match:
                    for m in match:
                        transfered = transfer_literal(m, mapping_struct)
                        if transfered:
                            new_clause.append(literal_to_str(transfered))
                if len(new_clause):
                    #struct[1][key] = ', '.join(new_clause)
                    new_nodes[key] = ', '.join(new_clause)
                else:
                    remove_nodes.append(key)
        struct[1] = new_nodes
    return copied