import csv
import pandas as pd
import numpy as np
soc_list = []
init_soc = .7
delta_t = 1/60
e_max = 10000
current_in = 0

df1 = pd.read_csv("current2.csv", header=0, names=['current'], dtype="float")
df2 = pd.read_csv("voltage2.csv", header=0, names=['voltage'], dtype="float")
#df3 = pd.read_csv("soc_30m.csv", header=0, names=['soc'], dtype="float")
#df3 = pd.read_csv("power2.csv", header=0, names=['power'], dtype="float")
#df3["power"]=df3["power"] * 1000


with open("soc_prediction.csv", "w") as fs:

#init
    #if df2['voltage'].iloc[0] != 0:
    #    current_in = df3['soc'].iloc[0] / df2['voltage'].iloc[0]
    #else:
    #    current_in = 0
    soc = init_soc + (((current_in-df1['current'].iloc[0]) * df2['voltage'].iloc[0]  * delta_t) / e_max)
    if soc > 1:
        soc = 1
    elif soc < 0:
        soc = 0
    print(soc)
    soc_list.append(soc)
    soc_wr = []
    soc_wr.append(soc)
    writer = csv.writer(fs, delimiter=',')
    writer.writerow(soc_wr)
    xper = 1
    print(df1.shape[0])#2049280
#get all next
    for x in range(df1.shape[0]):
        if x == 0:
            continue

        soc = soc_list[x-1] + (((current_in-df1['current'].iloc[x]) * df2['voltage'].iloc[x]  * delta_t) / e_max)
        if soc > 1:
            soc = 1
        elif soc < 0:
            soc = 0
        #print(soc)
        soc_list.append(soc)
        soc_wr = []
        soc_wr.append(soc)
        writer = csv.writer(fs, delimiter=',')
        writer.writerow(soc_wr)
    print("done")
