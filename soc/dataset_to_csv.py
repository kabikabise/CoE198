#f=open("household_power_consumption.txt", "r")
import csv
cline2 = []
with open("household_power_consumption.txt", "r") as fs1:
    with open("power2.csv", "w") as fs2:
        fs2.write("power\n")
        for line in fs1:
            cline = line.split(";")
            try:
                cline[2] = float(cline[2])
                cline2 = []
                cline2.append(cline[2])
                writer = csv.writer(fs2, delimiter=',')
                writer.writerow(cline2)
            except ValueError:
                pass


with open("household_power_consumption.txt", "r") as fs1:
    with open("current2.csv", "w") as fs2:
        fs2.write("current\n")
        for line in fs1:
            cline = line.split(";")
            try:
                cline[2] = float(cline[5])
                cline2 = []
                cline2.append(cline[5])
                writer = csv.writer(fs2, delimiter=',')
                writer.writerow(cline2)
            except ValueError:
                pass


with open("household_power_consumption.txt", "r") as fs1:
    with open("voltage2.csv", "w") as fs2:
        fs2.write("voltage\n")
        for line in fs1:
            cline = line.split(";")
            try:
                cline[2] = float(cline[4])
                cline2 = []
                cline2.append(cline[4])
                writer = csv.writer(fs2, delimiter=',')
                writer.writerow(cline2)
            except ValueError:
                pass

#fs2.write(cline[2]+"\r")
