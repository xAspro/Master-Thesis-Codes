import numpy as np
from astropy.cosmology import FlatLambdaCDM
import sys
import os

filename = 'CEERS_Kocevski.txt'
output_filename = '../Data_2019+/CEERS_Kocevski_sample.dat'
area = 0.009583
sample_number = 26         # Couldnt find samples more than 14, so instead of 15, going for 25. Just in case, dont want to overwrite anything.
zmin, zmax = 32, 37
Zmagmin, Zmagmax = 57, 62
def pcondition(z):
    return 1                    
max_entry_flag = True
max_entry = 1
header = '# Data from CEERS\n# K-corr is not mentioned in the paper. And K-corr formula isnt matching with the M1450 given. Need negative 2ish K corr.\n# But why wouldnt they directly give the K-corr if its from same survey? Can I take K_corr such that it works for 1?\n# Considering Area to be total area 14000\n# Taking probability from paper Kocevski et al. (2023)\n# counter  z     M1450  p       area    sample'



data=[]

with open(filename, 'r') as file:
    for line in file:
        if max_entry_flag:
            if max_entry <= len(data):
                break
        if line[0] == '#':
            continue
        print(f"line[zmin:zmax]: _{line[zmin:zmax]}_")
        print(f"line[Zmagmin:Zmagmax]: _{line[Zmagmin:Zmagmax]}_")
        z = float(line[zmin:zmax])  # z 
        Zmag = float(line[Zmagmin:Zmagmax])  # Zmag 

        print(f"z: {z} Zmag: {Zmag}")
        # sys.exit()
        data.append([z, Zmag, pcondition(z), area, sample_number])

print(data)

data = np.array(data)

indexed_data = np.column_stack((np.arange(1, data.shape[0] + 1), data))


with open(output_filename, 'w') as f:
    f.write(header + "\n")
    for row in indexed_data:
        f.write("{:9}{:7.3f}{:7.2f}{:8.5f}{:9.2f}  {:2d}\n".format(int(row[0]), row[1], row[2], row[3], row[4], int(row[5])))

print(f"Data written to {output_filename}")