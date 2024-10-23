import subprocess

# List of commands to run
commands = [
    "python selmap2.py 0 croom09ngp 2SLAQ_NGP",
    "python selmap2.py 0 croom09sgp 2SLAQ_SGP",
    "python selmap2.py 0 dr3z2p6 SDSS_DR3",
    "python selmap2.py 0 dr7z2p2 SDSS_DR7",
    "python selmap2.py 0 dr7z3p7 SDSS_DR7",
    "python selmap2.py 0 jiang16main SDSS_Main",
    "python selmap2.py 0 jiang16overlap SDSS_Main",
    "python selmap2.py 0 jiang16s82 SDSS_Main",
    # "python selmap2.py 0 mcgreer13_dr7",
    # "python selmap2.py 0 mcgreer13_s82",
]

# Run each command
for command in commands:
    print(f"Running command: {command}")
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
    else:
        print("Command executed successfully")