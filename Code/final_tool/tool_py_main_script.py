############ MAIN PYTHON SCRIPT ############
## This script include all functions needed to perform
## genome wide prediction of new data


###### Import packages ######

# General
import numpy as np
import pandas as pd
import random
from collections import Counter
from datetime import datetime

# Plotting
# import matplotlib as mpl
# import matplotlib.pylab as pl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.image as mpimg
# import seaborn as sns  
# import itertools

# Path and drive
import os 
import tempfile

# Modeling
import lightgbm as lgb
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model


# CV and preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Scoring
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc


# General evaluation functions 

def get_tnr_tpr(ytrue, ypred, digit=4):
    """
    Compute TNR and TPR.

    :ytrue: vector of true labels
    :ypred: vector of predicted labels

    Optional:
    :digit: int, number of decimal digits 
    """
    cf_matrix = confusion_matrix(ytrue, ypred)
    group_counts = [f"{value}" for value in cf_matrix.flatten()]
    row_sums = cf_matrix.sum(axis=1)
    norm_matrix = cf_matrix / row_sums[:, np.newaxis]
    group_percentages = [round(value, digit) for value in norm_matrix.flatten()]
    tnr, tpr = group_percentages[0], group_percentages[3]
    return tnr, tpr


def get_eval(ytrue, ypred):
    """
    Compute F1-score, MCC, TNR, and TPR.

    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """
    f1 = f1_score(ytrue, ypred, average='macro')
    mcc = matthews_corrcoef(ytrue, ypred)
    tnr, tpr = get_tnr_tpr(ytrue, ypred)
    return f1, mcc, tnr, tpr


def print_eval(ytrain, ytrain_pred, ytest, ytest_pred, is_test=False, print_train=True):
    """
    Print F1-score, MCC, TNR, TPR, precision, and recall.

    :ytrain: vector of train samples true labels
    :ytrain_pred: vector of train samples predicted labels
    :ytest: vector of test samples true labels
    :ytest_pred: vector of test samples predicted labels

    Optional:
    :is_test: boolean
    :print_train: boolean
    """  
    if is_test:  
        name = "Test"
    else:
        name = "Val"
    if print_train:
        train_f1, train_mcc, _, _ = get_eval(ytrain, ytrain_pred)
    print(f"Train F1-score: {train_f1:.4}")  
    print(f"Train MCC: {train_mcc:.4}")
    test_f1, test_mcc, test_tnr, test_tpr = get_eval(ytest, ytest_pred)
    print(f"{name} F1-score: {test_f1:.4}")
    print(f"{name} MCC: {test_mcc:.4}")
    print(f"{name} TNR = {test_tnr*100:.2f}%")
    print(f"{name} TPR = {test_tpr*100:.2f}%")
    print(f"{name} Precision = {precision_score(ytest, ytest_pred):.4}")
    print(f"{name} Recall = {recall_score(ytest, ytest_pred):.4}")


# Functions for heatmap evaluation plot 

# def evaluate(ml_output):
#     """
#     Compute accuracy, F1-score, and MCC (used in the performance evaluation heatmap). 

#     :ml_output: panda dataframe including yprob, ypred, ytrue (returned by lgb_cv, cnn_cv, or final_pred())
#     """
#     acc = accuracy_score(ml_output["ytrue"], ml_output["ypred"])
#     f1 = f1_score(ml_output["ytrue"], ml_output["ypred"], average='macro')
#     mcc = matthews_corrcoef(ml_output["ytrue"], ml_output["ypred"])
#     return acc, f1, mcc

   
# def evaluate_all(ml_output, digit=2):
#     """
#     Compute accuracy, ROC-AUC, F1-score, MCC, TNR, TPR, and precision 
#     (used in the performance evaluation heatmap). 

#     :ml_output: panda dataframe including "yprob", "ypred", "ytrue" (returned by lgb_cv, cnn_cv, or final_pred())

#     Optional:
#     :digit: int, number of decimal digits 
#     """
#     acc, f1, mcc = evaluate(ml_output)
#     tnr, tpr = get_tnr_tpr(ml_output["ytrue"], ml_output["ypred"], digit)
#     auc = roc_auc_score(ml_output["ytrue"], ml_output["ypred"])
#     precision = precision_score(ml_output["ytrue"], ml_output["ypred"])
#     return acc, auc, f1, mcc, tnr, tpr, round(precision, digit)  


# def eval_all_to_df(ml_output_lst, index=None, digit=2):
#     """
#     Compute accuracy, ROC-AUC, F1-score, MCC, TNR, TPR, and precision 
#     for each element of the input list. It returns a dataframe with the 
#     computed metrics (used in the performance evaluation heatmap). 

#     :ml_output_lst: a list of ml_outputs (panda dataframes including yprob, ypred, ytrue)

#     Optional:
#     :index: None or list of string including the name of each "ml_output"
#     :digit: int, number of decimal digits 
#     """
#     df = pd.DataFrame(columns=["F1", "MCC", "TNR", "TPR", "Prec"])   # "Acc", "AUC", 
#     for ml_output in ml_output_lst:
#         acc, auc, f1, mcc, tnr, tpr, precision = evaluate_all(ml_output, digit=digit)
#         df = df.append({"F1": f1, "MCC": mcc, "TNR": tnr, "TPR": tpr, "Prec": precision}, # "Acc", "AUC", 
#                         ignore_index=True)
#     if index is not None:
#         df["Method"] = index
#         df = df.set_index("Method")
#     return df


# def eval_heatmap(ml_output_lst, global_out_path, global_filename,                             
#                  index=None, decimal=2, cbar=True, figsize=(3.9, 6), 
#                  shrink=1, ticks=[.5, .6, .7, .8, .9], vmin=0.5, vmax=1, 
#                  title="", filename="", save=False, cmap="YlGnBu"):
#     """
#     Plot an heatmap showing ROC-AUC, F1-score, MCC, TNR, TPR, and precision 
#     for each element of the input list.

#     :ml_output_lst: a list of "ml_outputs" (panda dataframes including yprob, ypred, ytrue)

#     Optional
#     :index: None or list of string including the name of each "ml_output"
#     :decimal: int, number of decimal digits 
#     :cbar: boolean
#     :figsize: tuple of int or float
#     :shrink: int or float, shrink the color bar
#     :ticks: list of int or float
#     :vmin, vmax: values to anchor the colormap 
#     :title: string
#     :filename: string
#     :save: boolean, if True it save the plot in global_out_path directory as .png
#     :cmap: string
#     """  
#     # Generate an evaluation df
#     df = eval_all_to_df(ml_output_lst, index=index, digit=decimal)
#     # Plot the heatmap
#     fig, ax = plt.subplots(figsize=figsize)

#     heatmap = sns.heatmap(df, cmap=cmap, vmin= vmin, vmax=vmax, linecolor="black",
#                         linewidth=0.3, cbar_kws=dict(shrink=shrink, ticks=ticks), #square=True,
#                         annot=True, cbar=cbar, fmt=f".{decimal}f")
#     heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0) 
#     plt.ylabel("")
#     plt.title(title, fontsize=15, pad=15)
#     plt.tick_params(labelbottom = False, bottom=False, top = False, labeltop=True)
#     plt.xticks(weight='bold')
#     plt.yticks(weight='bold')
#     if save:
#         plt.savefig(f"{global_out_path}heatmap_performance_{filename}_{global_filename}.png", 
#                     dpi=300, bbox_inches="tight")
#     plt.show()


# Get number of samples in each class

def get_no_samples(ytrain, ytest=None, return_train_only=True, verbose=1):
    """
    Compute the number of samples of each class. If verbose > 0, 
    it also print it on the screen.

    :ytrain: vector of train samples true labels

    Optional:
    :ytest: None or dataframe, training data
    :return_train_only: boolean
    :verbose: int
    """
    neg_train, pos_train = np.bincount(ytrain)
    total_train = neg_train + pos_train
    if ytest is not None:
        neg_test, pos_test = np.bincount(ytest)
        total_test = neg_test + pos_test
    if verbose > 0:
        print(f"\nTrain examples:\n    Total: {total_train}\n    Positive: {pos_train} ({100 * pos_train / total_train:.2f}% of total)\n")
    if ytest is not None:
        print(f"Val/Test examples:\n    Total: {total_test}\n    Positive: {pos_test} ({100 * pos_test / total_test:.2f}% of total)\n")
    if return_train_only:
        return total_train, neg_train, pos_train
    else:
        return total_train, neg_train, pos_train, total_test, neg_test, pos_test


# Apply optimal binary classification threshold

def apply_threshold_pred(ml_output, threshold):
    """
    Apply a given classification threshold to the predictions 
    obtained by a machine learning model ("ml_output").

    :ml_output: panda dataframe including yprob, ypred, ytrue (returned by lgb_cv, cnn_cv, or final_pred())
    """  
    ml_output = ml_output.copy()
    ml_output["ypred"] = ml_output["yprob"].apply(lambda x: 1 if x > threshold else 0) 
    return ml_output


# Prediction of new data 

def final_pred(xtest, ytest, metadata, scaler_model_name, 
               global_model_path, global_filename_model, 
               algo="cnn", n_iter=7, 
               norm=True, plt_save=False): 
    """
    Function used to perform the prediction of new data.

    The function uses the StandardScaler and the machine learning models
    (LGBM or CNN) saved during the training process (one for each CV iteration). 
    For each CV iteration that have been used to perform the training, the new 
    data is normalized (if "norm" is set to True) and the loaded model is used 
    to perform the prediction. The final prediction is obtained by averaging the 
    probabilities predicted by the models obtained at each CV iteration. 
    If the true labels ("ytest") are provided, the function also show the 
    model performance showing a summary text and generating an heatmap.
    The final output is a pandas dataframes including the results obtained: 
    "yprob", "ypred", and "ytrue" (NaN if "ytest" is set to None).

    :xtest: dataframe, data to predict
    :ytest: None or vector of true labels
    :metadata: pandas dataframe with metadata information
    :scaler_model_name: string (e.g. "CNN_7folds" or "LGBM_7folds"), name of the models

    Optional
    :algo: string ("cnn" or "lgbm"), machine learning algorithm
    :filename: string, root of the model filename
    :model_path: string, path of the model directory
    :norm: boolean, if True fit, apply and save a StandardScaler
    :plt_save: boolean, if True it save the final plot in global_out_path directory as .png
    """  

    # Initialize   
    start_time = datetime.now()
    print(f"Performing prediction with {scaler_model_name} model")   
    if ytest is not None:
        total_test, neg_test, pos_test = get_no_samples(ytest, verbose=0)
        print(f"Test examples:\n    Total: {total_test}\n    Positive: {pos_test} ({100 * pos_test / total_test:.2f}% of total)") 
    else:
        ytest = [np.nan for i in range(len(xtest))]
    yprob_vec = np.array([]).reshape(-1,1)
    output_lst = []
    
    # Perform prediction with each saved model
    for i in range(n_iter):  
        print(f"\n> Starting pred iteration {i+1}") 

        # Load the scaler 
        name = f"{scaler_model_name}_cv_{i+1}"
        scaler_filepath = f"{global_model_path}{name}_scaler_{global_filename_model}.pkl" 
        print(f"Loading scaler:\t{scaler_filepath}")  
        scaler = pickle.load(open(scaler_filepath, "rb")) 

        # Normalize
        i_xtest = xtest.copy()
        if norm:
            i_xtest = pd.DataFrame(scaler.transform(i_xtest))  
        
        # Load the model
        model_filepath = f"{global_model_path}{name}_model_{global_filename_model}"
        if algo == "cnn":
            model_filepath = f"{model_filepath}.h5"
            print(f"Loading model:\t{model_filepath}")
            model = load_model(model_filepath)  
            # Rehape
            i_xtest = i_xtest.values.reshape((i_xtest.shape[0], -1, 1))
        if algo == "lgbm":
            model_filepath = f"{model_filepath}.txt"
            print(f"Loading model:\t{model_filepath}")
            model = lgb.Booster(model_file = model_filepath)
        
        # Pred
        yprob = model.predict(i_xtest).reshape(i_xtest.shape[0])
        ypred = np.round(yprob)
        # Store pred
        i_output = pd.DataFrame({"yprob": yprob, "ypred": ypred,"ytrue": ytest})
        # Eval
        if not np.isnan(ytest).any():
            print("Report:")
            print_eval(None, None, ytest, ypred, is_test=True, print_train=False)
        
        # Store iter predictions  
        if i == 0:
            axis = 0
        else:
            axis = 1
        yprob_vec = np.concatenate((yprob_vec, yprob.reshape(-1,1)), axis=axis)
        output_lst.append(i_output)
    
    # Final predictions
    yprob_final = np.mean(yprob_vec, axis=1)
    ypred_final = np.round(yprob_final)
    final_output = pd.DataFrame({"yprob": yprob_final, "ypred": ypred_final,"ytrue": ytest})
    # Final report
    if not np.isnan(ytest).any():
        print(f"\n>> {name} pred final report")
        print_eval(None, None, ytest, ypred_final, is_test=True, print_train=False)
        # # Final plot
        # names_index = [f"{algo.upper()} {i+1}" for i in range(n_iter)]
        # names_index.append("Final model")
        # output_lst.append(final_output)
        # eval_heatmap(output_lst, index=names_index, title=f"{algo.upper()} Test", filename= f"{scaler_model_name}_test", 
        #             save=plt_save, vmin=0.5, vmax=0.9, figsize=(4.7, 3.5))
    # Report time 
    print(f"Duration: {datetime.now() - start_time}")
    return final_output


