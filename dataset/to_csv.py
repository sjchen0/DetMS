def address_to_csv(read_path, save_path):
    with open(read_path, "r") as f:
        lines = f.readlines()
    with open(save_path, "w") as g:
        g.write("addrID,addrHash\n")
        for line in lines:
            g.write(line.replace(' ', ','))

def tx_to_csv(rp, sp):
    with open(rp, "r") as f:
        lines = f.readlines()
    with open(sp, "w") as g:
        g.write("txID,blockID,n_inputs,n_outputs,btime\n")
        for line in lines:
            g.write(line.replace(' ',','))

def txinout_to_csv(rp, sp):
    with open(rp, "r") as f:
        lines = f.readlines()
    with open(sp, "w") as g:
        g.write("txID,addrID,value\n")
        for line in lines:
            g.write(line.replace(' ',','))

if __name__ == "__main__":
    rt = "../../archive/"
    ds1 = "dataset1_2014_11_1500000/"
    ds2 = "dataset2_2015_6_1500000/"
    ds3 = "dataset3_2016_1_1500000/"
    for ds in [ds1,ds2,ds3]:
        address_to_csv(rt+ds+"addresses.txt", rt+ds+"addresses.csv")
        tx_to_csv(rt+ds+"tx.txt", rt+ds+"tx.csv")
        txinout_to_csv(rt+ds+"txin.txt", rt+ds+"txin.csv")
        txinout_to_csv(rt+ds+"txout.txt", rt+ds+"txout.csv")
