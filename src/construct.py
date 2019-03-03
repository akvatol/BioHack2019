
import numpy as np
import networkx as nx
import h5py
import time
import os
from itertools import combinations
import gc
import argparse as ag

from mpi4py import MPI

def time(func):
    import time

    def wrapper(*args, **kwargs):
        t = time.clock()
        res = func(*args, **kwargs)
        tdelta = time.clock() - t
        print(tdelta)
        return res, tdelta

    return wrapper

class APGraph(nx.DiGraph):
    
    def all_vs_all_paths(self, out_file, decode_dic, custom=False):
        self.my_tpsort = list(nx.topological_sort(self))
        self.my_order = {k: i for i, k in enumerate(self.my_tpsort)}
        self.my_ordered_pairs = []
        for i, p1 in enumerate(self.my_tpsort[:-1]):
            self.my_ordered_pairs += [(p1, p2) for p2 in self.my_tpsort[i+1:]]
        
        self.my_all_paths = {}
        fid = open(out_file, 'w')
        
        if custom:
            raise ValueError('Custom method is not implemented yet')
        else:
            for i, pair in enumerate(self.my_ordered_pairs):
                curr_paths = list(nx.all_simple_paths(self, *pair))
                
                if len(curr_paths) == 0:
                    continue
                else:
                    self.my_all_paths[pair] = curr_paths
                    for _path in curr_paths:
                        weights = [self.get_edge_data(_path[i], _path[i+1])['weight']\
                                   for i in range(len(_path)-1)]
                        fid.write('{}, {}, {}, {} \n'\
                        .format(*pair, [decode_dic[x] for x in _path], weights))
            
        fid.close()

'''
    def get_paths(self, source, sink):
        
        start_i = order[source]
        finish_i = order[sink]
        
        assert(finish_i == tp_sort[finish_i])
        
        paths = {k: [] for k in self.tpsort[start_i:finish_i]}
        paths[sink] = [[sink]]
        n_paths = {k: 0 for k in self.tpsort[start_i:finish_i]}
        n_paths[sink] = 1
        
        for v in self.tpsort[finish_i:start_i:-1]:
            succesors = list(self.neighbors(v))
            for succ in successors:
                n_paths[v] += n_paths[succ]
                for path in paths[succ]:
                    pass #to be continued
                
        return paths, n_paths, source, sink

     
def make_copy_graph(Graph, temp_path, copy_type=APGraph):
    with open(temp_path, 'wb') as fn:
        nx.write_edgelist(Graph, fn)
    New_Graph = nx.read_edgelist(temp_path, nodetype=int, create_using=copy_type)
    os.system(f"rm {temp_path}")
    return New_Graph


def corr_if(dic, pair):
    try:
        return dic[pair]
    except:
        return dic[pair[::-1]]
'''

###PARSER CLASS###
    
class Parse_to_Graph:
    
    def __init__(self, path: str, db_wkeys):
        self.G = APGraph()
        self.file = path
        db = h5py.File('12x3.hdf5', 'r')
        
        from collections import defaultdict
        self.indices = defaultdict(list)
        i = 0
        for k in db.keys():
            for sub_k in db[k].keys():
                self.indices[k+sub_k] = i
                i+=1
        self.inv_indices = {v: k for k, v in self.indices.items()}

    def line_parser(self, string: str):
        new_str = string.split(',')[:5]
        node1 = new_str[0] + new_str[1]
        node2 = new_str[2] + new_str[3]

        pair = new_str[4]
        
        if self.weighted:
            weight = len(pair)//2
        else:
            weight = 1

        if int(pair[0:len(pair) // 2]) > int(pair[len(pair) // 2:]):
            self.G.add_edge(self.indices[node1], self.indices[node2], weight=weight)
        else:
            self.G.add_edge(self.indices[node2], self.indices[node1], weight=weight)

    def construct_graph(self, weighted=True):
        
        self.weighted = weighted
        
        with open(self.file, 'r') as fn:
            for line in fn:
                self.line_parser(line)
        self.flag = True
        self.is_acyclic = nx.is_directed_acyclic_graph(self.G)
        return self.G, self.inv_indices
    
    def longest_peptide(self):
        if self.flag and self.is_acyclic:
            lpath = nx.dag_longest_path(self.G)
            lpep = ''
            [print(self.inv_indices[el], el) for el in lpath]
            
            for i in range(0, len(lpath)-1, 2):
                weight = self.G.get_edge_data(lpath[i], lpath[i+1])['weight']
                lpep += (self.inv_indices[lpath[i]]+self.inv_indices[lpath[i+1]][weight:])
            return lpep        
        else:
            raise ValueError("Graph haven't been yet constructed or it contains cycles")
            
    def find_components(self):
        self.components = nx.weakly_connected_component_subgraphs(self.G)
        return list(self.components)
    
def getArgs():
    
    parser = ag.ArgumentParser()

    parser.add_argument('-i', '--input-file',required=True, dest='input_files',type=str, help='Input file with peptides')
    parser.add_argument('-db', '--databse-file',required=True,dest='db',type=str, help='Input database')
    parser.add_argument('-odir','--output-directory', required=True, dest='out',type=str, help='Destination file')
    parser.add_argument('-of', '--master-file', required=True, dest='MF', type=str, help='Master file name')
    #parser.add_argument('njobs', '--number-of-cores', type=str, help='Number of cores to run') 
    
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = getArgs()
    
    input_peptides = args['input_files']
    input_db = args['db']
    out_dir = args['out']
    MF = args['MF']
    
    if os.path.isdir(out_dir):
        if os.listdir(out_dir):
            raise ValueError('OUTPUT DIRECTORY IS NOT EMPTY')
    else:
        os.system(f'mkdir {out_dir}')
    
    comm = MPI.COMM_WORLD
    
    inv_ind = None
    WCCs = None
    
    if comm.rank == 0:
        parse_instance = Parse_to_Graph(input_peptides, input_db)
        _, inv_ind = parse_instance.construct_graph() #
        WCCs = list(parse_instance.find_components())
        #WCCs = sorted(WCCs, key=lambda x: x.number_of_nodes(), reverse=True)
        from random import shuffle
        shuffle(WCCs)

    inv_ind = comm.bcast(inv_ind)
    WCCs = comm.bcast(WCCs)
    
    i = comm.rank
    while i < len(WCCs):
        
        wpath = os.path.join(out_dir, f'{i}.txt')
        WCCs[i].all_vs_all_paths(wpath, inv_ind)
        i += comm.size
        
    if comm.rank == 0:
        
        os.system(f'cat {out_dir}/* > {MF}')
    
