import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import ipdb
from collections import defaultdict
import networkx as nx

def load_data(ds_num, cfg):
    ds_path = cfg["ds{}_path".format(ds_num)]
    addr_data_path = ds_path + "addresses.csv"
    tx_data_path = ds_path + "tx.csv"
    tx_in_data_path = ds_path + "txin.csv"
    tx_out_data_path = ds_path + "txout.csv"
    addr_data = pd.read_csv(addr_data_path)
    tx_data = pd.read_csv(tx_data_path)
    tx_in_data = pd.read_csv(tx_in_data_path)
    tx_out_data = pd.read_csv(tx_out_data_path)
    ret = dict(
        addr_data=addr_data, 
        tx_data=tx_data, 
        tx_in_data=tx_in_data, 
        tx_out_data=tx_out_data
    )
    return ret

def build_single_aain_and_tain(addr_data, tx_data, tx_in_data, tx_out_data, start_time, duration=3600, build_tain=False):
    print("filtering data within time slot...")
    t1 = time.time()
    tx_data_filtered = tx_data.query('btime >= {} & btime <= {}'.format(start_time, start_time + duration))
    tx_data_filtered = tx_data_filtered[["txID", "btime"]]
    dict_to_btime = pd.Series(tx_data_filtered.btime.values, index=tx_data_filtered.txID).to_dict()
    tx_data_filtered_np = tx_data_filtered.to_numpy()
    tx_in_data_filtered = tx_in_data[tx_in_data["txID"].isin(tx_data_filtered_np[:,0])]
    tx_out_data_filtered = tx_out_data[tx_out_data["txID"].isin(tx_data_filtered_np[:,0])]
    tx_in_data_filtered_np = tx_in_data[tx_in_data["txID"].isin(tx_data_filtered_np[:,0])].to_numpy()
    tx_out_data_filtered_np = tx_out_data[tx_out_data["txID"].isin(tx_data_filtered_np[:,0])].to_numpy()
    t2 = time.time()
    # ipdb.set_trace()
    print("finished, using {:.2f} sec.".format(t2-t1))
    print("start building AAIN...")
    # data structure of aain: addr_from, addr_to, txID, time
    aain = []
    for txID in tqdm(tx_data_filtered_np[:,0]):
        '''
        General rules for creating the weighted DiGraph:
        Situations can be that addrs_from is empty, or addrs_to is empty. These transactions are ignored. 
        If a transaction is multiple-input-multiple-output, we measure the weight between any involving pair of addresses by weighted averaging.
        Let {1,2,...,n} be sending addresses and {1,2,...,m} be receiving addresses. 
        Each sending address contributes v_i to the total transaction amount V = sum v_i, 
        and each receiving address contributes v_j. Observe that V ~= sum v_j due to the service charges.
        Then, weight[i,j] := v_j * v_i / V.
        '''
        df_txin_txID = tx_in_data_filtered.loc[tx_in_data_filtered["txID"] == txID]
        df_txout_txID = tx_out_data_filtered.loc[tx_out_data_filtered["txID"] == txID]
        V = df_txin_txID["value"].sum()
        a_v_txin_txID = df_txin_txID
        a_v_txout_txID = df_txout_txID
        if df_txin_txID.duplicated(subset=["addrID"]).any():
            a_v_txin_txID = df_txin_txID.groupby(["addrID"], as_index=False)["value"].sum()
        if df_txout_txID.duplicated(subset=["addrID"]).any():
            a_v_txout_txID = df_txout_txID.groupby(["addrID"], as_index=False)["value"].sum()
        addrs_from = a_v_txin_txID["addrID"].to_numpy()
        vs_from = a_v_txin_txID["value"].to_numpy()
        addrs_to = a_v_txout_txID["addrID"].to_numpy()
        vs_to = a_v_txout_txID["value"].to_numpy()
        if len(addrs_from) == 0 or len(addrs_to) == 0:
            continue
        btime = dict_to_btime[txID]
        values = (vs_from[:,None] @ vs_to[None, :] / V).reshape((-1,))
        aain.append(np.concatenate((np.array(np.meshgrid(addrs_from, addrs_to, txID, btime)).T.reshape(-1,4), values[:,None]), axis=1))
        '''
        [[a1,a1]
         [a1,a2]
         [a1,a3]
         ...
         [a2,a1]
         [a2,a2]
         [a2,a3]
         ...
         [an,am]]
        '''
    aain = np.concatenate(aain)
    aain_df = pd.DataFrame(aain, columns=['addr_from', 'addr_to', 'txID', 'time', 'weight'])
    aain_df = aain_df.astype({'addr_from': int, 'addr_to': int, 'txID': int, 'time': int})
    # check parallel edges
    # for addr_from in pd.unique(aain_df['addr_from']):
    #     match = aain_df.loc[aain_df['addr_from'] == addr_from]
    #     if match['addr_to'].duplicated().any():
    #         print(match)
    aain_G = nx.from_pandas_edgelist(aain_df, "addr_from", "addr_to", ["weight", "time"], create_using=nx.DiGraph())
    # ipdb.set_trace()
    tain = defaultdict(dict)
    if build_tain:
        # data structure of tain: {addrID: {incoming: [txIDs, btimes, values], outgoing: [txIDs, btimes, values]}}
        print("start building TAIN...")
        in_dict = tx_in_data_filtered.groupby("addrID").groups
        out_dict = tx_out_data_filtered.groupby("addrID").groups
        for addrID in tqdm(in_dict.keys()):
            txIDs = tx_in_data_filtered["txID"][in_dict[addrID]].to_numpy()
            btimes = np.array([dict_to_btime[t] for t in txIDs])
            values = tx_in_data_filtered["value"][in_dict[addrID]].to_numpy()
            tain[addrID]["outgoing"] = np.array([txIDs, btimes, values])
        for addrID in tqdm(out_dict.keys()):
            txIDs = tx_out_data_filtered["txID"][out_dict[addrID]].to_numpy()
            btimes = np.array([dict_to_btime[t] for t in txIDs])
            values = tx_out_data_filtered["value"][out_dict[addrID]].to_numpy()
            tain[addrID]["incoming"] = np.array([txIDs, btimes, values])
    
    # ipdb.set_trace()
    return aain, aain_G, tain

def count_motif(addr_data, aain, tain):
    pass



if __name__ == "__main__":
    root_path="/root/Challenge/Vincent/ESTR3108/archive/"
    cfg = dict(
        ds1_path=root_path+"dataset1_2014_11_1500000/",
        ds2_path=root_path+"dataset2_2015_6_1500000/",
        ds3_path=root_path+"dataset3_2016_1_1500000",
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
