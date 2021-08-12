#!/usr/bin/env python3

############ SCRIPT TO PREDICTION NEW DATA FOR FINAL TOOL ############
## This script perform the prediction of new data
## and it is used in the final tool implemented in this project

print("\nRUNNING PROFILES PREDICTION")

#### Impor module ####
from tool_py_main_script import *
import argparse

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="input directory", type=str, default="./")
parser.add_argument("-f", "--filename", help="filename in input directory", type=str, default="filename")
parser.add_argument("-o", "--output", help="output directory", type=str, default="./")
parser.add_argument("-a", "--algorithm", help="machine learning algorithm, select 'cnn' or 'lgbm'", type=str, default="lgbm")
parser.add_argument("-M", "--model_dir", help="model and scaler directory", type=str, default="models/")
parser.add_argument("-m", "--model_name", help="model and scaler filename", type=str, default="fit_on_timepoint_0_to_2_processed")
args = parser.parse_args()


#### Initialize ####

# Input and output
print("\nPrediction initialization:")
# print("Input:", args.input_dir)
# print("Filename:", args.filename)
# print("Output:", args.output)
# print("Algorithm:", args.algorithm)
# print("Model and scaler directory:", args.model_dir)
# print("Model and scaler filename:", args.model_name)

INPUT = args.input_dir
FILENAME = args.filename
OUTPUT = args.output
MODEL_PATH = args.model_dir
FILENAME_MODEL = args.model_name

# Model and scaler                   
# MODEL_PATH = "/binf-isilon/alab/students/stefano/thesis_project/Models/pos_neg_shift_timepoint_0to2_all_merged_subtnorm/"           
# FILENAME_MODEL = "pos_neg_shift_timepoint_0to2_all_merged_subtnorm"       


#### Load data ####
print("\nLoading files..")
X_TEST = pd.read_csv(f"{INPUT}{FILENAME}/profiles_{FILENAME}_subtnorm.csv")
metadata_test = pd.read_csv(f"{INPUT}{FILENAME}/metadata_{FILENAME}_subtnorm.csv")
print("Shape profiles and metadata:", X_TEST.shape, metadata_test.shape)


#### Genome wide prediction ####

if args.algorithm == "lgbm":
    # LGBM
    output = final_pred(X_TEST, None, metadata_test, 
                            global_model_path=MODEL_PATH, 
                            global_filename_model=FILENAME_MODEL,  
                            algo="lgbm", scaler_model_name="LGBM_7folds", plt_save=True)

elif args.algorithm == "cnn":
    # CNN
    output = final_pred(X_TEST, None, metadata_test, 
                            global_model_path=MODEL_PATH, 
                            global_filename_model=FILENAME_MODEL,
                            algo="cnn", scaler_model_name="CNN_7folds5", plt_save=True)

else:
    raise NameError("algorithm must be 'cnn' or 'lgbm'")

output.to_csv(f"{OUTPUT}{FILENAME}/tres_prediction_{FILENAME}.csv", index=False)
