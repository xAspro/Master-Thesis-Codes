import os
import subprocess

# Define the path to the directory containing the data files
data_dir = 'Data_new'

# Define the path to the selmap.py script
selmap_script = 'selmap.py'

# Iterate over each file in the data directory
for filename in os.listdir(data_dir):
    # Construct the command to run selmap.py with the file as an argument
    command = ['python', selmap_script, filename]
    print("Running command:", command)
    
    # Run the command
    subprocess.run(command)
