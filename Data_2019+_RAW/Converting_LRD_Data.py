import numpy as np
import matplotlib.pyplot as plt

data = []
column = []

input_filename = 'LRD_Data_RAW.txt'
with open(input_filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('#'):
            column.append((line[2:]).strip())
            data.append([])
            continue
        data[-1].append(line.strip())

output_filename = '../Data_2019+/LRD_Data.txt'
with open(output_filename,'w') as f:
    f.write('# LRD Data from Krishna from Kocevski et al\n')
    f.write(f'# Reading columns from the file: {column}\n')
    f.write('# Name                z     M1450\n')
    redshifts = []
    for i in range(len(data[0])):
        f.write(f'{(data[0][i]+' '+data[1][i]):<20}  {data[2][i]:<4}  {data[3][i]:<4}\n')
        redshifts.append(float(data[2][i]))


# Plot the histogram of redshifts
plt.hist(redshifts, bins=20, edgecolor='black')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.title('Histogram of Redshifts')
plt.show()


# A = [1, 1, 1, 2, 3, 2, 3, 1, 1, 1, 5, 3, 1, 1, 3, 4, 7]
# unique_entries = np.unique(A)
# print(unique_entries)
# # Get the indices of all occurrences of the number 1 in A using numpy functions
# indices_of_ones = np.where(np.array(A) == 1)[0]
# print(indices_of_ones)
# for i in range(len(indices_of_ones)):
#     print(A[indices_of_ones[i]])


# mask = np.array(A) == 1
# print(mask)
# A = np.array(A)[mask]
# print(A)

# But using mask is better as it is faster due to vectorisation

# import sys
# sys.exit(0)
unique_Surveys = np.unique(data[0])


area_sq_armin = {
    'CEERS': 88.6,
    'JADES': 62.1,
    'NGDEEP': 11.4,
    'PRIMER-COS': 138.9,
    'PRIMER-UDS': 243.0,
    'UNCOVER': 43.8
}

sample = {
    'CEERS': 151,
    'JADES': 152,
    'NGDEEP': 153,
    'PRIMER-COS': 154,
    'PRIMER-UDS': 155,
    'UNCOVER': 156
}


for survey in unique_Surveys:
    out_file = '../Data_2019+/LRD_Data_'+str(survey)+'_sample.txt'
    mask = np.array(data[0]) == survey
    data_survey = np.array(data)[:, mask]

    # Convert the data_survey to a list of tuples for sorting
    data_survey_list = list(zip(data_survey[0], data_survey[1], data_survey[2].astype(float), data_survey[3].astype(float)))

    # Sort the list based on z value (index 2) and then M1450 value (index 3)
    data_survey_list.sort(key=lambda x: (x[2], x[3]))

    # Convert back to numpy array after sorting
    data_survey_sorted = np.array(data_survey_list)

    cnt = 1
    with open(out_file, 'w') as f:
        f.write('# LRD Data from Krishna from Kocevski et al\n')
        # f.write(f'# Reading columns from the file: {column}\n')
        f.write(f'# Survey: {survey}\n')
        f.write('# The value or p is taken to be 1 for now\n')
        f.write('# counter   z       M1450   p        area      sample\n')
        for i in range(len(data[0])):
            if data[0][i] == survey:
                z_value = float(data_survey_sorted[cnt-1][2])
                M1450_value = float(data_survey_sorted[cnt-1][3])
                f.write(f'  {cnt:<5}     {z_value:<6.2f}  {M1450_value:<6.2f}  1.00000  {(area_sq_armin[survey]/3600):<8.6f}  {sample[survey]:<5}\n')
                cnt += 1
    print(f'Written data to {out_file}')


    z_range = np.arange(data_survey_sorted[:, 2].astype(float).min()-1, data_survey_sorted[:, 2].astype(float).max()+1, 0.05)
    mag_range = np.arange(data_survey_sorted[:, 3].astype(float).min()-1, data_survey_sorted[:, 3].astype(float).max()+1, 0.05)

    selfunc_filename = '../Data_2019+/LRD_Data_'+str(survey)+'_selfunc.txt'
    with open(selfunc_filename, 'w') as f:
        f.write('# LRD Data from Krishna from Kocevski et al\n')
        f.write(f'# Survey: {survey}\n')
        f.write('# The value or p is taken to be 1 for now\n')
        f.write('# counter   z       M1450   p\n')

        cnt = 1
        for i in range(len(z_range)):
            for j in range(len(mag_range)):
                f.write(f'  {cnt:<5}     {z_range[i]:<6.2f}  {mag_range[j]:<6.2f}  1.00000\n')
                cnt += 1
    print(f'Written data to {selfunc_filename}')
