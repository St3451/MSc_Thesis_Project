#!/usr/bin/env python3

############ SIMPLE SCRIPT TO GENERATE DIRECTORIES ############
## This script generate the directories needed to
## store the results obtained by the tool


# Import
import os
import argparse

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="filename in input directory", type=str, default="filename")
parser.add_argument("-o", "--output", help="output directory", type=str, default="./")
args = parser.parse_args()

OUTPUT = args.output
FILENAME = args.filename

# Create directories
paths = [f"{OUTPUT}{FILENAME}", 
         f"{OUTPUT}{FILENAME}/plots",
         f"{OUTPUT}{FILENAME}/plots/all_profiles",
         f"{OUTPUT}{FILENAME}/plots/predicted_profiles"]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
