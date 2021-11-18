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
import json
from build_aain import *
from collections import defaultdict
import random
import pickle

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

    if not cfg["directed"]:
        nx_G = nx_G.to_undirected()
    G = node2vec.Graph(nx_G, cfg["directed"], cfg["p"], cfg["q"])
    G.preprocess_transition_probs()
    walks = G.simulate_walks(cfg["num_walks"], cfg["walk_length"])
    features = learn_embeddings(walks)
    # import ipdb; ipdb.set_trace()
    return features


def down_sample(walks, num_walks):
    '''
    ensure number of walks starting from each node = num_walks
    '''
    walk_dict = dict()
    for walk in walks:
        if walk[0] not in walk_dict.keys():
            walk_dict[walk[0]] = [walk]
        else:
            walk_dict[walk[0]].append(walk)
    for node in walk_dict.keys():
        if len(walk_dict[node]) > num_walks:
            random.shuffle(walk_dict[node])
            walk_dict[node] = walk_dict[node][:num_walks]
    ret = []
    for w in walk_dict.values():
        ret += w
    # import ipdb; ipdb.set_trace()
    return ret

if __name__ == "__main__":
    with open("../../../config/config.json", "r") as f:
        cfg = json.load(f)
        cfg["ds1_path"] = cfg["root_path"] + cfg["ds1_path"]
        cfg["ds2_path"] = cfg["root_path"] + cfg["ds2_path"]
        cfg["ds3_path"] = cfg["root_path"] + cfg["ds3_path"]
    print("loading csv data...")
    t1 = time.time()
    data_dict = load_data(1, cfg)
    t2 = time.time()
    print("finished, using {:.2f} sec.".format(t2-t1))
    addr_data = data_dict["addr_data"]
    tx_data = data_dict["tx_data"]
    ds_begin_time = tx_data["btime"].min()
    print(ds_begin_time)
    ds_end_time = tx_data["btime"].max()
    print(ds_end_time)
    tx_in_data = data_dict["tx_in_data"]
    tx_out_data = data_dict["tx_out_data"]
    start_time = ds_begin_time
    duration = cfg['snapshot_duration']
    walks = []
    snapshot_cnt = 0
    while start_time + duration <= ds_end_time:
        print("start_time: {}, duration: {}".format(start_time, duration))
        aain, aain_G, tain = build_snapshot(addr_data, tx_data, tx_in_data, tx_out_data, start_time, duration=duration)
        if not cfg["directed"]:
            aain_G = aain_G.to_undirected()
        G = node2vec.Graph(aain_G, cfg["directed"], cfg["p"], cfg["q"])
        G.preprocess_transition_probs()
        walks_in_snapshot = G.simulate_walks(cfg["num_walks"], cfg["walk_length"])
        walks += walks_in_snapshot
        start_time += duration
        snapshot_cnt += 1
    # import ipdb; ipdb.set_trace()
    with open("walks_temp.pkl", "rb") as f:
        pickle.dump(walks, f)
    # with open("walks_temp.pkl", "rb") as f:
    #     walks = pickle.load(f)
    down_sample(walks, cfg["num_walks"])
    print("start learning embeddings...")
    features = learn_embeddings(walks)
    with open("features_temp.pkl", "rb") as g:
        pickle.dump(features, g)
    # import ipdb; ipdb.set_trace()
