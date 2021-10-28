'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import sys
import os
sys.path.append("../../../dataset")
import time
from build_aain import *

def read_graph():
    '''
        Reads the input network in networkx.
        '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()
    return G

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=cfg["dimensions"], window=cfg["window_size"], min_count=0, sg=1, workers=cfg["workers"], iter=cfg["iter"])
    features = dict()
    for node in model.wv.vocab.keys():
        features[node] = model.wv[node]
    return features

def learn(nx_G, cfg):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    G = node2vec.Graph(nx_G, cfg["directed"], cfg["p"], cfg["q"])
    G.preprocess_transition_probs()
    walks = G.simulate_walks(cfg["num_walks"], cfg["walk_length"])
    features = learn_embeddings(walks)
    # import ipdb; ipdb.set_trace()
    return features

if __name__ == "__main__":
    root_path="/root/Challenge/Vincent/ESTR3108/archive/"
    cfg = dict(
        ds1_path=root_path+"dataset1_2014_11_1500000/",
        ds2_path=root_path+"dataset2_2015_6_1500000/",
        ds3_path=root_path+"dataset3_2016_1_1500000",
        dimensions=128,
        walk_length=80,
        num_walks=10,
        window_size=10,
        iter=1,
        workers=8,
        p=1,
        q=1,
        weighted=True,
        directed=True,
    )
    print("loading csv data...")
    t1 = time.time()
    data_dict = load_data(1, cfg)
    t2 = time.time()
    print("finished, using {:.2f} sec.".format(t2-t1))
    addr_data = data_dict["addr_data"]
    tx_data = data_dict["tx_data"]
    ds_begin_time = tx_data["btime"].min()
    ds_end_time = tx_data["btime"].max()
    tx_in_data = data_dict["tx_in_data"]
    tx_out_data = data_dict["tx_out_data"]
    aain, aain_G, tain = build_single_aain_and_tain(addr_data, tx_data, tx_in_data, tx_out_data, 1414771239)
    features = learn(aain_G, cfg)
    # import ipdb; ipdb.set_trace()