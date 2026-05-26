
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

def preparation():

    print("Starting the preparation function...")
    file_path = r"C:\Users\lenao\Project_emulator_python\inputs\fdsFit_Martini1pi_2021.root"
    root_file = uproot.open(file_path)

    systematics = [
        'Flux Systematics',
        'ND280 Detector Systematics',
        'Cross-Section Systematics',
        'Cross-Section (binned) Systematics'
    ]

    samples = ['FHC FGD1 #nu_{#mu} CC 0#pi 0p 0#gamma', 'FHC FGD1 #nu_{#mu} CC 0#pi Np 0#gamma']
    variables = ['Pmu', 'CosThetamu']
    reactions = ['1;1', '2;1']

    return systematics, samples, variables, reactions