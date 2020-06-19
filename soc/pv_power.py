import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
from math import sqrt
import statistics
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
soc_list = []
init_soc = .7
delta_t = 1/360
e_max = 10000
r = 0.2
rated_cap = 300* 15
temp_c = -0.004
dhi_stc = 1000
temp_stc = 26
fudge = 20
pv_list = []
pv_pred_list = []
pv_act_list = []
pv_pred_list_trunc = []
pv_act_list_trunc = []

df1 = pd.read_csv("current_30m.csv", header=0, names=['current'], dtype="float")
df2 = pd.read_csv("voltage_30m.csv", header=0, names=['voltage'], dtype="float")
df3 = pd.read_csv("soc_30m.csv", header=0, names=['soc'], dtype="float")
df4 = pd.read_csv("dhi.csv", header=0, skiprows=[1, 2], names=['time', 'value', 'step'])
df5 = pd.read_csv("temp.csv", header=0, skiprows=[1, 2], names=['time', 'value', 'step'])

with open("pv_prediction_30m.csv", "w") as fs:
    alpha = temp_c * (abs(df5['step'].iloc[0]+fudge - 25))
    pv_power = rated_cap * (df4['step'].iloc[0]/dhi_stc) * (1 + alpha)
    pv_power = abs(pv_power)
    pv_list = []
    pv_list.append(pv_power)
    print(pv_power)
    for x in range(df4.shape[0]):
        alpha = temp_c * (abs(df5['step'].iloc[x]+fudge  - 25))
        pv_power = rated_cap * (df4['step'].iloc[x]/dhi_stc) * (1 + alpha)
        pv_power = abs(pv_power)
        pv_list = []

        pv_pred_list.append(pv_power)
        pv_list.append(pv_power)
        writer = csv.writer(fs, delimiter=',')
        writer.writerow(pv_list)

with open("pv_actual_30m.csv", "w") as fs:
    alpha = temp_c * (abs(df5['value'].iloc[0]+fudge - 25))
    pv_power = rated_cap * (df4['value'].iloc[0]/dhi_stc) * (1 + alpha)
    pv_power = abs(pv_power)
    pv_list = []
    pv_list.append(pv_power)
    print(pv_power)
    for x in range(df4.shape[0]):
        alpha = temp_c * (abs(df5['value'].iloc[x]+fudge  - 25))
        pv_power = rated_cap * (df4['value'].iloc[x]/dhi_stc) * (1 + alpha)
        pv_power = abs(pv_power)
        pv_list = []
        pv_act_list.append(pv_power)
        pv_list.append(pv_power)
        writer = csv.writer(fs, delimiter=',')
        writer.writerow(pv_list)


#truncate
pv_pred_list_trunc = pv_pred_list[6000:6500]
pv_act_list_trunc = pv_act_list[6000:6500]
q75, q25 = np.percentile(pv_pred_list_trunc, [75,25])
iqr = q75-q25

rmse = rmse(np.array(pv_act_list_trunc), np.array(pv_pred_list_trunc))
print("rmse: " + str(rmse))
nrmse = rmse / iqr
print("nrmse: " + str(nrmse))
plt.plot(df4.index,pv_pred_list, label='predicted')
plt.plot(df4.index,pv_act_list, label='actual')

plt.legend(loc=3, bbox_to_anchor=(1,0))
plt.xlim([6000,6200])
plt.ylim([0, 4000])
plt.ylabel("PV Power (W)")
plt.xlabel("Time(hr)")
plt.savefig('pv_power.png', bbox_inches="tight")
plt.show()
