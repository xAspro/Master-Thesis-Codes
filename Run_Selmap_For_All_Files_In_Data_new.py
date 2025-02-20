import os
import re
import subprocess

# Define the path to the directory containing the data files
data_dir = 'Data_new'

# Define the path to the selmap.py script
selmap_script = 'selmap.py'

# # Iterate over each file in the data directory
# for filename in os.listdir(data_dir):
#     # Construct the command to run selmap.py with the file as an argument
#     command = ['python', selmap_script, filename]
#     print("Running command:", command)
    
#     # Run the command
#     subprocess.run(command)


# Create lists to store the different types of files
sample_dat_files = []
selfunc_dat_files = []
other_files = []


# Regular expression for matching 'sel.dat', 'sel1.dat', 'sel2.dat', 'sel_4.dat', 'sel_correction.dat', etc.
sel_dat_pattern = re.compile(r'.*sel.*\.dat')


# Iterate over each file in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith('sample.dat'):
        sample_dat_files.append(filename)
    elif filename.endswith('selfunc.dat') or sel_dat_pattern.match(filename):
        selfunc_dat_files.append(filename)
    else:
        other_files.append(filename)

# Sort the lists
sample_dat_files.sort()
selfunc_dat_files.sort()
other_files.sort()

print("Files with 'sample' versions:")
for sample_file in sample_dat_files:
    print(sample_file)
print("\n\nFiles with 'selfunc' versions:")
for selfunc_file in selfunc_dat_files:
    print(selfunc_file)
print("\n\nFiles with neither 'sample' nor 'selfunc' versions:")
for other_file in other_files:
    print(other_file)

# Find files that have both 'sample' and 'selfunc' versions
common_files = []
for sample_file in sample_dat_files:
    base_name = sample_file.replace('sample.dat', '')
    corresponding_selfunc_file = base_name + 'selfunc.dat'
    if corresponding_selfunc_file in selfunc_dat_files:
        common_files.append(base_name)

# Write the results to an output file
output_file = 'Run_Selmap_summary.txt'
with open(output_file, 'w') as f:
    f.write("Files with both 'sample' and 'selfunc' versions:\n")
    for base_name in common_files:
        f.write(f"{base_name}sample.dat\t\t{base_name}selfunc.dat\n")
    
    f.write("\n\n\nFiles with only 'sample' versions:\n")
    for sample_file in sample_dat_files:
        base_name = sample_file.replace('sample.dat', '')
        if base_name not in common_files:
            f.write(f"{sample_file}\n")
    
    f.write("\n\n\nFiles with only 'selfunc' versions:\n")
    for selfunc_file in selfunc_dat_files:
        base_name = selfunc_file.replace('selfunc.dat', '')
        if base_name not in common_files:
            f.write(f"{selfunc_file}\n")
    
    f.write("\n\n\nFiles with neither 'sample' nor 'selfunc' versions:\n")
    for other_file in other_files:
        f.write(f"{other_file}\n")

print(f"Summary written to {output_file}")
