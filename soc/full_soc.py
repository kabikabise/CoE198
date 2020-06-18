import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
soc_list = []
pv_list = []
load = []
init_soc = .7
delta_t = 1/60
e_max = 10000
rated_cap = 300
temp_c = -0.004
dhi_stc = 1000
temp_stc = 26

df1 = pd.read_csv("current_30m.csv", header=0, names=['current'], dtype="float")
df2 = pd.read_csv("voltage_30m.csv", header=0, names=['voltage'], dtype="float")
df3 = pd.read_csv("soc_30m.csv", header=0, names=['soc'], dtype="float")
df4 = pd.read_csv("dhi.csv", header=0, skiprows=[1, 2], names=['time', 'value', 'step'])
df5 = pd.read_csv("temp.csv", header=0, skiprows=[1, 2], names=['time', 'value', 'step'])
df6 = pd.read_csv("pv_prediction_30m.csv", header=0, names=['pv'], dtype="float")

with open("soc_prediction_30m.csv", "w") as fs:
#init
    if df2['voltage'].iloc[0] != 0:
        current_in = df6['pv'].iloc[0] / df2['voltage'].iloc[0]
    else:
        current_in = 0
    soc = init_soc + (((current_in-df1['current'].iloc[0]) * df2['voltage'].iloc[0]  * delta_t) / e_max)
    if soc > 1:
        soc = 1
    elif soc < 0:
        soc = 0
    print(soc)
    soc_list.append(soc)
    pv_list.append(current_in)
    loadval = df1['current'].iloc[0] * df2['voltage'].iloc[0]
    load.append(loadval)
    soc_wr = []
    soc_wr.append(soc)
    writer = csv.writer(fs, delimiter=',')
    writer.writerow(soc_wr)
    xper = 1
    print(df1.shape[0])#2049280
#get all next
    for x in range(df6.shape[0]):
        if x == 0:
            continue

        if df2['voltage'].iloc[x] != 0:
            current_in = df6['pv'].iloc[x] / df2['voltage'].iloc[x]
        else:
            current_in = 0
        soc = soc_list[x-1] + (((current_in-df1['current'].iloc[x]) * df2['voltage'].iloc[x]  * delta_t) / e_max)
        if soc > 1:
            soc = 1
        elif soc < 0:
            soc = 0
        #print(soc)
        soc_list.append(soc)
        pv_list.append(current_in)
        loadval = df1['current'].iloc[x] * df2['voltage'].iloc[x]
        load.append(loadval)
        soc_wr = []
        soc_wr.append(soc)
        writer = csv.writer(fs, delimiter=',')
        writer.writerow(soc_wr)
    print("done")
#plt.figure()
f, axes = plt.subplots(3,1)
axes[0].plot(df6.index,soc_list)
axes[0].set_ylabel('SoC')
axes[0].set_xlim([100, 300])
axes[0].set_ylim([0,0.5])
axes[0].grid()
axes[0].set_xlabel('Time(hr)')

axes[1].plot(df6.index,pv_list)
axes[1].set_ylabel('PV current(A)')
axes[1].set_xlim([100, 300])
axes[1].set_ylim([0,20])
axes[1].grid()
axes[1].set_xlabel('Time(hr)')

axes[2].plot(df6.index,load)
axes[2].set_ylabel('Load(W)')
axes[2].set_xlim([100, 300])
axes[2].set_ylim([0,6000])
axes[2].grid()

axes[2].set_xlabel('Time(hr)')


#plt.plot(df6.index,load, label='soc')
#plt.plot(df6.index,pv_list, label='pv current')
#plt.plot(df6.index,load, label='load')
#plt.legend(loc=3, bbox_to_anchor=(1,0))
#plt.xlim([1, 100])
#plt.ylim([0, 4000])
#plt.ylabel("SoC, PV and Load")
#plt.grid()
#plt.xlabel("Time(hr)")
#plt.show()
plt.savefig('soc_graph.png')
