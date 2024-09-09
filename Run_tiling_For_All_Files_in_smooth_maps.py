import os
import subprocess
import sys

overwrite = sys.argv[1] # '0' or '1'
# Define the path to the directory containing the data files
data_dir = 'smooth_maps'

# Define the path to the selmap.py script
selmap_script = 'tiling.py'

file_count = 0
# Iterate over each file in the data directory
for filename in os.listdir(data_dir):
    if filename.endswith('_with_tiles.dat'):
        continue
    if not overwrite:
        if os.path.exists(os.path.join(data_dir, filename[:-4]+'_with_tiles.dat')):
            continue
    file_count += 1
    print()
    # Construct the command to run selmap.py with the file as an argument
    command = ['python', selmap_script, filename]

    print("Running command:", command)
    print("File count:", file_count)

    # Run the command
    subprocess.run(command)
