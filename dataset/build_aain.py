import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import ipdb
from collections import defaultdict

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

def build_single_aain_and_tain(addr_data, tx_data, tx_in_data, tx_out_data, start_time, duration=3600):
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
    print("finished, using {:.2f} sec.".format(t2-t1))
    print("start building AAIN...")
    # data structure of aain: addr_from, addr_to, txID, time
    aain = []
    for txID in tqdm(tx_data_filtered_np[:,0]):
        addrs_from = tx_in_data_filtered_np[tx_in_data_filtered_np[:,0] == txID][:,1]
        addrs_to = tx_out_data_filtered_np[tx_out_data_filtered_np[:,0] == txID][:,1]
        btime = dict_to_btime[txID]
        aain.append(np.array(np.meshgrid(addrs_from, addrs_to, txID, btime)).T.reshape(-1,4))
    aain = np.concatenate(aain)
    # print(aain)
    
    print("start building TAIN...")
    tain = defaultdict(dict)
    # data structure of tain: {addrID: {incoming: [txIDs, btimes, values], outgoing: [txIDs, btimes, values]}}
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
    return aain, tain

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
    build_single_aain_and_tain(addr_data, tx_data, tx_in_data, tx_out_data, 1414771239)