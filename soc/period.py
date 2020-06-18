import csv
import pandas as pd
import numpy as np
period = 30
with open("soc_prediction_1m.csv", "r") as fs1:
    with open("soc_30m.csv", "w") as fs2:
        fs2.write("soc\n")
        count = 0
        for line in fs1:
            if count%30 == 0:
                cline = line.split(",")
                try:
                    cline[0] = float(cline[0])
                    cline2 = []
                    cline2.append(cline[0])
                    writer = csv.writer(fs2, delimiter=',')
                    writer.writerow(cline2)
                except ValueError:
                    pass
                count = 1
            else:
                count = count + 1

with open("current2.csv", "r") as fs1:
    with open("current_30m.csv", "w") as fs2:
        fs2.write("current\n")
        count = 0
        for line in fs1:
            if count%30 == 0:
                cline = line.split(",")
                try:
                    cline[0] = float(cline[0])
                    cline2 = []
                    cline2.append(cline[0])
                    writer = csv.writer(fs2, delimiter=',')
                    writer.writerow(cline2)
                except ValueError:
                    pass
                count = 1
            else:
                count = count + 1

with open("voltage2.csv", "r") as fs1:
    with open("voltage_30m.csv", "w") as fs2:
        fs2.write("voltage\n")
        count = 0
        for line in fs1:
            if count%30 == 0:
                cline = line.split(",")
                try:
                    cline[0] = float(cline[0])
                    cline2 = []
                    cline2.append(cline[0])
                    writer = csv.writer(fs2, delimiter=',')
                    writer.writerow(cline2)
                except ValueError:
                    pass
                count = 1
            else:
                count = count + 1
