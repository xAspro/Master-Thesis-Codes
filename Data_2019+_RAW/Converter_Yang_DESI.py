import numpy as np
from astropy.cosmology import FlatLambdaCDM
import sys
import os

filename = 'DESI_main_selection.txt'
output_filename = '../Data_2019+/DESI_main_selection.txt'
K_corr = 2.5  # Assuming K-correction of 2.5
area = 14000
sample_number = 25         # Coulding find samples more than 14, so instead of 15, going for 25. Just in case, dont want to overwrite anything.
zmin, zmax = 47, 51
Zmagmin, Zmagmax = 52, 57
def pcondition(z):
    if z > 6.4:
        p = 0.05
    elif z > 5.7:
        p = 0.08
    elif 4.8 < z < 5.4:
        p = 0.39
    else:
        p = 0.23    #I dont know why, but there is a missing gap. I will take the average in that case.
    return p
header = '# Data from DESI_main_selection\n# Assumed K-correction to be 2.5\n# Considering Area to be total area 14000\n# Taking probability from paper\n# counter  z     M1450  p       area    sample'



# filename = 'DESI_SV1.txt'
# output_filename = '../Data_2019+/DESI_SV1.txt'
# K_corr = 2.5  # Assuming K-correction of 2.5
# area = 14000
# sample_number = 25         # Coulding find samples more than 14, so instead of 15, going for 25. Just in case, dont want to overwrite anything.
# zmin, zmax = 45, 49
# Zmagmin, Zmagmax = 50, 55
# def pcondition(z):
#     if z > 6.4:
#         p = 0.05
#     elif z > 5.7:
#         p = 0.08
#     elif 4.8 < z < 5.4:
#         p = 0.39
#     else:
#         p = 0.23    #I dont know why, but there is a missing gap. I will take the average in that case.
#     return p
# header = '# Data from DESI_SV1\n# Assumed K-correction to be 2.5\n# Considering Area to be total area 14000\n# Taking probability from paper\n# counter  z     M1450  p       area    sample'


data=[]

with open(filename, 'r') as file:
    for line in file:
        if line[0] == '#':
            continue
        z = float(line[zmin:zmax])  # z 
        Zmag = float(line[Zmagmin:Zmagmax])  # Zmag 
        # Perform your calculations with z and Zmag

        # print(f"z: {z} Zmag: {Zmag}")
        # sys.exit()
        data.append([z, Zmag, pcondition(z)])

# print(data)


# Define the cosmology 
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

data2=[]

for i in range(len(data)):
    # print("data[i]: ",data[i])
    z, Zmag, p = data[i]
    # Calculate M1450 using the distance modulus formula
    M1450 = Zmag - 5 * np.log10(cosmo.luminosity_distance(z).to('pc').value) + 5 - K_corr

    # print(f"Absolute magnitude (M1450): {M1450:.2f}")

    data2.append([z, M1450, p, area, sample_number])


data = np.array(data2)
sorted_data = data[data[:, 0].argsort()]
indexed_data = np.column_stack((np.arange(1, sorted_data.shape[0] + 1), sorted_data))


with open(output_filename, 'w') as f:
    f.write(header + "\n")
    for row in indexed_data:
        f.write("{:9}{:7.3f}{:7.2f}{:8.5f}{:9.2f}  {:2d}\n".format(int(row[0]), row[1], row[2], row[3], row[4], int(row[5])))

print(f"Data written to {output_filename}")

# np.savetxt(output_filename, indexed_data, header=header, fmt=('%d', '%04.3f', '%04.2f', '%06.5f', '%06.2f', '%d'))


# import numpy as np

# # Generating example data
# data = np.random.rand(5, 3)  # Replace with your actual data
# print(data)
# # Sorting the data based on the first column
# sorted_data = data[data[:, 0].argsort()]

# # Adding an index column
# indexed_data = np.column_stack((np.arange(1, sorted_data.shape[0] + 1), sorted_data))

# # Define your header
# # header = "This file contains data for XYZ\nIndex Column1 Column2 Column3"
# print(indexed_data)
# # Save the sorted and indexed data to the file
# # np.savetxt('output.txt', indexed_data, header=header, comments='', fmt='%d %.6f %.6f %.6f')
