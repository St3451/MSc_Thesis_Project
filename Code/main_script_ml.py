############ MAIN SCRIPT ############
## This script include all functions needed to train, evaluate and use  
## the machine learning framework implemented for the thesis project    


###### Import packages ######

# General
import numpy as np
import pandas as pd
import random
from collections import Counter
from datetime import datetime

# Plotting
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import seaborn as sns
import itertools

# Path and drive
import os 
import tempfile

# Modeling
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model

import shap

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
from sklearn.metrics import log_loss

# Tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt import gp_minimize
from skopt.space import Real
from bayes_opt import BayesianOptimization
from hyperopt import STATUS_OK, hp, fmin, tpe, Trials



###### General evaluation functions ######

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


def get_auc_fpr_tpr(ml_output, return_threshold=False):
    """
    Compute ROC-AUC, FPR, TPR.

    :ml_output: pandas dataframe including yprob, ypred, ytrue (returned by lgb_cv() or cnn_cv())

    Optional:
    :return_threshold: boolean, if True it also returns a vector of 
    classification thresholds used to build the ROC 
    """
    roc_auc = roc_auc_score(ml_output["ytrue"], ml_output["ypred"])
    fpr, tpr, threshold = roc_curve(ml_output["ytrue"], ml_output["yprob"])
    if return_threshold:
        return roc_auc, fpr, tpr, threshold
    else:
        return roc_auc, fpr, tpr


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


def print_output_eval(train_output, test_output, is_test=False):
    """
    Print F1-score, MCC, TNR, TPR, precision, and recall. 
    It is simply a pharser for the print_eval function.

    :train_output: pandas dataframe including yprob, ypred, ytrue (training data)
    :test_output: pandas dataframe including yprob, ypred, ytrue (test or validation data)

    Optional:
    :is_test: boolean
    """  
    print_eval(train_output["ytrue"].values, train_output["ypred"].values, 
               test_output["ytrue"].values, test_output["ypred"].values, is_test)



###### Final evaluation functions ######

def plot_clf_report(train_output, test_output, name="", 
                    font_scale=1.3, title_font_size=15, 
                    figsize=(14, 4), save=False):
    """
    Plot the scikit-learn classification_report as heatmap.

    :train_output: pandas dataframe including yprob, ypred, ytrue (train data)
    :test_output: pandas dataframe including yprob, ypred, ytrue (test or validation data)

    Optional:
    :name: string
    :font_scale: int or float
    :title_font_size: int or float
    :figsize: tuple of int or float
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    """
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2)
    sns.set(font_scale=font_scale, style="ticks")
    # Train
    ax = plt.subplot(gs[0, 0])
    clf_report = classification_report(train_output["ytrue"], train_output["ypred"],
                                        labels=[0,1], output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].drop(["weighted avg"], axis=1).T, 
                annot=True, linewidths=1, cmap="viridis", ax=ax, vmin=0.6, vmax=0.95)
    ax.set_title(f"{name} train CR", fontsize=title_font_size)
    # Test/Validation
    ax = plt.subplot(gs[0, 1])
    clf_report = classification_report(test_output["ytrue"], test_output["ypred"],
                                        labels=[0,1], output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].drop(["weighted avg"], axis=1).T, 
                annot=True, linewidths=1, cmap="viridis", ax=ax, vmin=0.6, vmax=0.95)
    ax.set_title(f"{name} test CR", fontsize=title_font_size)
    # Layout and save
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUT_PATH}classification_report_{FILENAME}_{name}.png", dpi = 300)
    plt.show()


def plot_confusion_matrix(ml_output_lst, names=None, 
                          font_scale=1.2, title_font_size=15, figsize=(10, 3),   
                          vmax=0.6, save=False, filename="", cmap='viridis'):
    """
    Plot a confusion matrix for each element of the input list (minimum two elements). 

    :ml_output_lst: List of "ml_outputs" (panda dataframes including yprob, ypred, ytrue)

    Optional:
    :names: None or list of string
    :font_scale: int or float
    :title_font_size: int or float
    :figsize: tuple of int or float
    :vmax: upper bound value to anchor the colormap 
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :filename: string
    :cmap: string
    """
    # Initialize
    if names is None:    
        names = "Train", "Val"
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(ml_output_lst))
    sns.set(font_scale=font_scale, style="ticks")
    # Plot train and test/validation 
    for i, output in enumerate(ml_output_lst):
        ax = plt.subplot(gs[0, i])
        # Get confusion matrix and labels
        cf_matrix = confusion_matrix(output["ytrue"], output["ypred"])
        group_names = ["True Neg","False Pos","False Neg","True Pos"]
        group_counts = [f"{value}" for value in cf_matrix.flatten()]
        row_sums = cf_matrix.sum(axis=1)
        norm_matrix = cf_matrix / row_sums[:, np.newaxis]
        group_percentages = [f"{value*100:.2f}%" for value in norm_matrix.flatten()]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        # Plot heatmap
        sns.heatmap(norm_matrix, annot=labels, annot_kws={"size": 15}, 
                    fmt="", vmin=0, vmax=1, cmap=cmap, linewidths=1, ax=ax)
        plt.ylabel("True", fontsize = 13)
        plt.xlabel("Predicted", fontsize = 13, labelpad = -0)
        plt.title(f"{names[i]}", fontsize = title_font_size)
    if save:
        plt.savefig(f"{OUT_PATH}confusion_matrix_{names}{filename}_{FILENAME}.png", 
                dpi = 300, bbox_inches="tight")
    plt.show()


## Functions for heatmap evaluation plot ##

def evaluate(ml_output):
    """
    Compute accuracy, F1-score, and MCC (used in the performance evaluation heatmap). 

    :ml_output: panda dataframe including yprob, ypred, ytrue (returned by lgb_cv, cnn_cv, or final_pred())
    """
    acc = accuracy_score(ml_output["ytrue"], ml_output["ypred"])
    f1 = f1_score(ml_output["ytrue"], ml_output["ypred"], average='macro')
    mcc = matthews_corrcoef(ml_output["ytrue"], ml_output["ypred"])
    return acc, f1, mcc


def evaluate_all(ml_output, digit=2):
    """
    Compute accuracy, ROC-AUC, F1-score, MCC, TNR, TPR, and precision 
    (used in the performance evaluation heatmap). 

    :ml_output: panda dataframe including "yprob", "ypred", "ytrue" (returned by lgb_cv, cnn_cv, or final_pred())

    Optional:
    :digit: int, number of decimal digits 
    """
    acc, f1, mcc = evaluate(ml_output)
    tnr, tpr = get_tnr_tpr(ml_output["ytrue"], ml_output["ypred"], digit)
    auc = roc_auc_score(ml_output["ytrue"], ml_output["ypred"])
    precision = precision_score(ml_output["ytrue"], ml_output["ypred"])
    return acc, auc, f1, mcc, tnr, tpr, round(precision, digit)  


def eval_all_to_df(ml_output_lst, index=None, digit=2):
    """
    Compute accuracy, ROC-AUC, F1-score, MCC, TNR, TPR, and precision 
    for each element of the input list. It returns a dataframe with the 
    computed metrics (used in the performance evaluation heatmap). 

    :ml_output_lst: a list of ml_outputs (panda dataframes including yprob, ypred, ytrue)

    Optional:
    :index: None or list of string including the name of each "ml_output"
    :digit: int, number of decimal digits 
    """
    df = pd.DataFrame(columns=["F1", "MCC", "TNR", "TPR", "Prec"])   # "Acc", "AUC", 
    for ml_output in ml_output_lst:
        acc, auc, f1, mcc, tnr, tpr, precision = evaluate_all(ml_output, digit=digit)
        df = df.append({"F1": f1, "MCC": mcc, "TNR": tnr, "TPR": tpr, "Prec": precision}, # "Acc", "AUC", 
                        ignore_index=True)
    if index is not None:
        df["Method"] = index
        df = df.set_index("Method")
    return df


def eval_heatmap(ml_output_lst, index=None, decimal=2, cbar=True, figsize=(3.9, 6), 
                 shrink=1, ticks=[.5, .6, .7, .8, .9], vmin=0.5, vmax=1, 
                 title="", filename="", save=False, cmap="YlGnBu"):
    """
    Plot an heatmap showing ROC-AUC, F1-score, MCC, TNR, TPR, and precision 
    for each element of the input list.

    :ml_output_lst: a list of "ml_outputs" (panda dataframes including yprob, ypred, ytrue)

    Optional
    :index: None or list of string including the name of each "ml_output"
    :decimal: int, number of decimal digits 
    :cbar: boolean
    :figsize: tuple of int or float
    :shrink: int or float, shrink the color bar
    :ticks: list of int or float
    :vmin, vmax: values to anchor the colormap 
    :title: string
    :filename: string
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :cmap: string
    """  
    # Generate an evaluation df
    df = eval_all_to_df(ml_output_lst, index=index, digit=decimal)
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=figsize)

    heatmap = sns.heatmap(df, cmap=cmap, vmin= vmin, vmax=vmax, linecolor="black",
                        linewidth=0.3, cbar_kws=dict(shrink=shrink, ticks=ticks), #square=True,
                        annot=True, cbar=cbar, fmt=f".{decimal}f")
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0) 
    plt.ylabel("")
    plt.title(title, fontsize=15, pad=15)
    plt.tick_params(labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    if save:
        plt.savefig(f"{OUT_PATH}heatmap_performance_{filename}_{FILENAME}.png", 
                    dpi=300, bbox_inches="tight")
    plt.show()


## Barplots accuracy, F1, MCC ##

def evaluate_lst(ml_output_lst):
    """
    Evaluate a list of predictions ("ml_output") returning three lists 
    (list of accuracy, list of F1-scores, list of MCC), where 
    each element correspond to the evaluation of the corresponding
    prediction.

    :ml_output_lst: a list of "ml_outputs" (panda dataframes including "yprob", "ypred", "ytrue") 
    """  

    lst_acc, lst_f1, lst_mcc = [], [], []
    for ml_output in ml_output_lst: 
        acc, f1, mcc = evaluate(ml_output)
        lst_acc.append(acc)
        lst_f1.append(f1)
        lst_mcc.append(mcc)
    return lst_acc, lst_f1, lst_mcc

def plot_metrics(train_output_lst, test_output_lst, names, 
                 bbox_to_anchor=(1.03, 0.6), annotation_size=13, 
                 save=False, figsize=(15,5), filename=""):
    """
    Generate a barplot showing F1-score, and MCC for each element of the input 
    lists ("train_output_lst", "test_output_lst", "names"). 

    :train_output_lst: a list of "ml_outputs" obtained on the train data 
    :train_output_lst: a list of "ml_outputs" obtained on the test data 
    :names: a list of names (corresponding to the provided "ml_outputs" lists)

    Optional
    :bbox_to_anchor: tuple of float or int, legend box position
    :annotation_size: int
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :figsize: tuple of int or float
    :filename: string
    """  

    # Group by metrics
    train_acc, train_f1, train_mcc = evaluate_lst(train_output_lst)
    test_acc, test_f1, test_mcc = evaluate_lst(test_output_lst)
    # Plot
    metric_df = pd.DataFrame({"Train F1-score": train_f1, "Train MCC": train_mcc,
                              "Val F1-score": test_f1, "Val MCC": test_mcc, }, 
                              index = names)
    ax = metric_df.plot(y=["Train F1-score", "Train MCC","Val F1-score", "Val MCC"], 
                        ylim=(0,1.1), figsize=figsize, kind="bar", zorder=3, ec ="black",
                        rot=1, width=0.8, color = ["lightsalmon", "tomato",
                                                    "skyblue", "cornflowerblue"])
    # Add details
    plt.title(f"F1-score and MCC ({filename.capitalize()})", fontsize = 15)
    plt.ylabel("Score")
    plt.grid(axis="y", zorder=0, color="lightgray")  
    legend = plt.legend(frameon = 1, shadow = True, bbox_to_anchor=bbox_to_anchor, fontsize=13)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    # Annotate scores on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        xpos='center'
        offset = {'center': 0, 'right': 1, 'left': -1}
        ax.annotate(f"{height:.2}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  
                    textcoords="offset points",  
                    ha=ha[xpos], va='bottom', size=annotation_size)
    if save:
        plt.savefig(f"{OUT_PATH}models_performance_{filename}_{FILENAME}.png", 
                    dpi = 300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()


## ROC and Precision-Recall ##

# ROC and log ROC
def plot_ROC(output_lst, names, plot_name="", save=False, filename="", bbox_to_anchor=(2.05, 1)):
    """
    Plot a ROC curve and a log ROC curve for each result in "output_lst".
    If "filename" is set to "all", CNN models will be shown with 
    a solid line and LGBM models with a dashed one.
    If "filename" is set to "train_val", performance on the training
    data will be indicated with a solid line and performance on the 
    training data with a dashed one.
    
    :output_lst: a list of "ml_outputs" (panda dataframes including "yprob", "ypred", "ytrue")
    :names: list of names
    
    Optional
    :plot_name: string
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :filename: string (set to "all" or "train_val" to have solid and dashed lines)
    :bbox_to_anchor: tuple of float or int, legend box position
    """  

    # Initialize
    fig, axes = plt.subplots(1, 2, figsize = (16, 4.5))
    colors = sns.color_palette().as_hex()
    # Duplicate colors to have corresponding color for LGBM and CNN
    if filename == "all" or filename == "train_val":
        colors = list(itertools.chain(*zip(colors,colors)))
    # ROC
    for i, output in enumerate(output_lst):
        # Dashed lines for LGBM or Train and solid for CNN or Val 
        ls = "-"
        if filename == "all" and names[i].split()[0] == "LGBM":
            ls = "--"
        elif filename == "train_val" and names[i].split()[1] == "Train":
            ls = "--"
        # Plot metrics
        roc_auc, fpr, tpr = get_auc_fpr_tpr(output)
        axes[0].plot(fpr, tpr, ls, label=f"{names[i]} (AUC = {roc_auc:.3})", zorder=3, c=colors[i], lw=0.8)   # lw=0.9
    axes[0].plot([0, 1], [0, 1],'r-.', lw=0.8)
    axes[0].set_title(f"ROC {plot_name}", fontsize=15)
    axes[0].set_xlabel('FPR (1 - Specificity)')
    axes[0].set_ylabel('TPR (Recall)')
    axes[0].grid(zorder=0, color="lightgray")
    # Log ROC
    for i, output in enumerate(output_lst):
        # Dashed lines for LGBM or Train and solid for CNN or Val 
        ls = "-"
        if filename == "all" and names[i].split()[0] == "LGBM":
            ls = "--"
        elif filename == "train_val" and names[i].split()[1] == "Train":
            ls = "--"
        # Plot metrics
        auc, fpr, tpr = get_auc_fpr_tpr(output)
        axes[1].plot(np.log10(fpr), tpr, ls, label=f"{names[i]} (AUC = {auc:.3})", zorder=3, c=colors[i], lw=0.8) # lw=0.9
    axes[1].set_title(f"log scale ROC {plot_name}", fontsize=15)
    axes[1].set_xlabel('log(FPR)')
    axes[1].set_ylabel('TPR')
    axes[1].grid(zorder=0, color="lightgray")
    # Details
    legend = plt.legend(fontsize=10, frameon = 1, 
                        shadow = True, bbox_to_anchor=bbox_to_anchor)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    fig.tight_layout()
    fig.subplots_adjust(top=0.89)
    if save:
        plt.savefig(f"{OUT_PATH}ROC_logROC_{plot_name}_{filename}_{FILENAME}.png", dpi = 300,
                    bbox_inches="tight")
    plt.show()


# ROC 
def plot_ROC_only(output_lst, names, plot_name="", save=False, filename="", bbox_to_anchor=(1.05, 1)):
    """
    Plot a ROC curve for each result in "output_lst".
    If "filename" is set to "all", CNN models will be shown with 
    a solid line and LGBM models with a dashed one.
    If "filename" is set to "train_val", performance on the training
    data will be indicated with a solid line and performance on the 
    training data with a dashed one.
    
    :output_lst: a list of "ml_outputs" (panda dataframes including "yprob", "ypred", "ytrue")
    :names: list of names
    
    Optional
    :plot_name: string
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :filename: string (set to "all" or "train_val" to have solid and dashed lines)
    :bbox_to_anchor: tuple of float or int, legend box position
    """  
    # Initialize
    fig, axes = plt.subplots(1, 1, figsize=(9.6, 4.2))
    colors = sns.color_palette().as_hex()
    # Duplicate colors to have corresponding color for LGBM and CNN
    if filename == "all" or filename == "train_val":
        colors = list(itertools.chain(*zip(colors,colors)))
    # ROC
    for i, output in enumerate(output_lst):
        # Dashed lines for LGBM or Train and solid for CNN or Val 
        ls = "-"
        if filename == "all" and names[i].split()[0] == "LGBM":
            ls = "--"
        elif filename == "train_val" and names[i].split()[1] == "Train":
            ls = "--"
        # Plot metrics
        roc_auc, fpr, tpr = get_auc_fpr_tpr(output)
        axes.plot(fpr, tpr, ls, label=f"{names[i]} (AUC = {roc_auc:.3})", zorder=3, c=colors[i], lw=0.8)   # lw=0.9
    axes.plot([0, 1], [0, 1], 'r-.', lw=0.8)
    axes.set_title(f"ROC {plot_name}", fontsize=15)
    axes.set_xlabel('FPR (1 - Specificity)')
    axes.set_ylabel('TPR (Recall)')
    axes.grid(zorder=0, color="lightgray")
    # Details
    legend = plt.legend(fontsize=10, 
                        frameon = 1, shadow = True, bbox_to_anchor=bbox_to_anchor)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    fig.tight_layout()
    fig.subplots_adjust(top=0.89)
    if save:
        plt.savefig(f"{OUT_PATH}ROC_{plot_name}_{filename}_{FILENAME}.png", dpi = 300,
                    bbox_inches="tight")
    plt.show()


# Precision-recall curve
def plot_precision_recall_curve(output_lst, names, plot_name="", save=False, filename="", 
                                legend_ncol=2, legend_outside=False, 
                                figsize=(6, 4.2), bbox_to_anchor=(2.05, 1)):
    """
    Plot a Precision-Recall curve for each result in "output_lst".
    If "filename" is set to "all", CNN models will be shown with 
    a solid line and LGBM models with a dashed one.
    If "filename" is set to "train_val", performance on the training
    data will be indicated with a solid line and performance on the 
    training data with a dashed one.
    
    :output_lst: a list of "ml_outputs" (panda dataframes including "yprob", "ypred", "ytrue")
    :names: list of names
    
    Optional
    :plot_name: string
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :filename: string (set to "all" or "train_val" to have solid and dashed lines)
    :legend_ncol: int, number of legend columns
    :legend_outside: boolean, if True place the legend outside the plotting frame
    :bbox_to_anchor: tuple of float or int, legend box position
    """    
        
    # Initialize
    plt.figure(figsize=figsize)
    colors = sns.color_palette().as_hex()
    # Duplicate colors to have corresponding color for LGBM and CNN
    if filename == "all" or filename == "train_val":
        colors = list(itertools.chain(*zip(colors,colors)))
    for i, test_output in enumerate(output_lst):
        # Dashed lines for LGBM or Train and solid for CNN or Val 
        ls = "-"
        if filename == "all" and names[i].split()[0] == "LGBM":
            ls = "--"
        elif filename == "train_val" and names[i].split()[1] == "Train":
            ls = "--"
        # Plot metrics
        precision, recall, thresholds = precision_recall_curve(test_output["ytrue"], test_output["yprob"])
        precision_recall_auc = auc(recall, precision)
        plt.plot(recall, precision, ls, label=f"{names[i]} (AUC = {precision_recall_auc:.3})", zorder=3, c=colors[i], lw=0.8)  # lw=0.9
    # Add details
    plt.grid(zorder=0, color="lightgray")
    plt.xlabel("TPR (Recall)")
    plt.ylabel("PPV (Precision)")
    plt.title(f"Precision-recall curve {plot_name}", fontsize=15)
    
    if legend_outside:
        legend = plt.legend(title="Models", fontsize=10.5, title_fontsize=13, loc = "lower left",
                            frameon = 1, shadow = True, ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor)
    else:
        legend = plt.legend(title="Models", fontsize=10.5, title_fontsize=13, loc = "lower left",
                            frameon = 1, shadow = True, ncol=legend_ncol)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    if save:
        plt.savefig(f"{OUT_PATH}precision-recall_curve_{plot_name}_{filename}_{FILENAME}.png", dpi = 300,
                    bbox_inches="tight")
    plt.show()


## Barplot of TNR and TPR ##

def get_tnr_tpr_lst(ml_output_lst, digit=2):
    """
    Compute the TNR and TPR of a list of predictions ("ml_output"). 
    It returns two lists (one for TNR and one for TPR) where 
    each element correspond to the evaluation of the corresponding
    prediction.

    :ml_output_lst: a list of "ml_outputs" (panda dataframes including "yprob", "ypred", "ytrue") 
    
    Optional
    :digit: int, number of decimal digits
    """   

    lst_tnr, lst_tpr = [], []
    for ml_output in ml_output_lst: 
        tnr, tpr = get_tnr_tpr(ml_output["ytrue"], ml_output["ypred"], digit)
        lst_tnr.append(tnr)
        lst_tpr.append(tpr)
    return lst_tnr, lst_tpr
  
def plot_tp_tn(train_output_lst, test_output_lst, names, 
               bbox_to_anchor=(1.03, 0.6), annotation_size=13, 
               digit=2, save=False, filename=""):
    """
    Generate a barplot showing TPR and TNR for each element of the input 
    lists ("train_output_lst", "test_output_lst", "names"). 
    
    :train_output_lst: a list of "ml_outputs" obtained on the train data 
    :train_output_lst: a list of "ml_outputs" obtained on the test data 
    :names: a list of names (corresponding to the provided "ml_outputs" lists)
    
    Optional
    :bbox_to_anchor: tuple of float or int, legend box position
    :annotation_size: int
    :digit: int, number of decimal digits
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :filename: string
    """   

    # Group by metrics
    train_tnr, train_tpr = get_tnr_tpr_lst(train_output_lst, digit)
    test_tnr, test_tpr = get_tnr_tpr_lst(test_output_lst, digit)
    # Plot
    metric_df = pd.DataFrame({"Train TNR": train_tnr, "Train TPR": train_tpr, 
                                "Val TNR": test_tnr, "Val TPR": test_tpr}, index = names)
    ax = metric_df.plot(y=["Train TNR", "Train TPR", "Val TNR", "Val TPR"], 
                        ylim=(0,1.1), figsize=(15,5), kind="bar", zorder=3, ec ="black",
                        rot=1, width=0.8, color = ["lightsalmon", "tomato", "skyblue", "cornflowerblue"])
    # Add details
    plt.title("TNR and TPR", fontsize = 17)
    plt.ylabel("Score", fontsize = 15)
    plt.grid(axis="y", zorder=0, color="lightgray")  
    legend = plt.legend(frameon = 1, shadow = True, bbox_to_anchor=bbox_to_anchor, fontsize=13)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    # Annotate scores on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        xpos='center'
        offset = {'center': 0, 'right': 1, 'left': -1}
        ax.annotate(f"{height}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  
                    textcoords="offset points",  
                    ha=ha[xpos], va='bottom', size=annotation_size)
    if save:
        plt.savefig(f"{OUT_PATH}tnr_tpr_barplot2_{filename}_{FILENAME}.png", 
                    dpi = 300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()



###### Preprocessing and other functions ######

# Normalization
def normalize(xtrain, xtest, save=False, name=""):
    """
    Use scikit-learn StandardScaler to standardize the data to 
    0 mean and 1 unit variance. 

    The scaler is fitted on the training data and it is applied 
    to both training and test.

    :xtrain: dataframe, training data
    :xtest: dataframe, test data 

    Optional:
    :save: boolean, if True the scaler is saved in MODEL_PATH directory as .pk1 file
    :name: root of the name for saving the scaler
    """
    scaler = StandardScaler()
    xtrain = pd.DataFrame(scaler.fit_transform(xtrain))
    # Save the scaler object
    if save:
        filepath = f"{MODEL_PATH}/{name}_scaler_{FILENAME}.pkl"    
        print(f"Saving scaler:\n\t{filepath}")
        pickle.dump(scaler, open(filepath,'wb'))    
    xtest = pd.DataFrame(scaler.transform(xtest))
    return xtrain, xtest

# CV split by chromosomes
def cv_split(metadata, n_val_chr=3, cv_seed=None):
    """
    Take as input a metadata dataframe and it output a list of 
    vector including the chromosomes that are used as validation in each
    CV iteration.

    :metadata: dataframe with metadata information

    Optional:
    :n_val_chr: int, number of validation chromosomes used at each fold
    :cv_seed: int or None
    """
    chr_list = np.unique(metadata.chr)
    random.seed(cv_seed)
    random.shuffle(chr_list)
    k = len(chr_list) // n_val_chr
    val_chr_folds = np.array_split(chr_list, k)
    return val_chr_folds

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


#  PCA plot
def multi_plot_pca(x, y, nrows=2,
                   xlim=(-1.5,1.5), ylim=(-1.5,1.5),
                   figsize=(12,7), top=0.89, targets = [1, 0],
                   title="", save=False, filename="pca_multiplot",
                   legend_anchor=(1.3, 1.25), legend_title=None, alpha=0.8):
    """
    PCA plot showing different principal components.

    :x: dataframe of samples 
    :y: vector of true labels
    
    Optional
    :nrows: int, if "nrows" > 2, it will show a larger number of PCs 
    :xlim: tuple of int or float
    :ylim: tuple of int or float
    :figsize: tuple of int or float
    :top: float or int, the position of the top edge of the subplots
    :targets: list or tuple of targets (e.g. classes, chromosomes, replicates etc)
    :title: string
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    :filename: string
    :legend_anchor: tuple of int or float, legend box position
    :legend_title: string
    :alpha: int or float
    """      
    # Perform PCA and MDS
    n_pcs = nrows*3+1
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(x)
    var = pca.explained_variance_ratio_ * 100
    tmp_df = pd.concat([pd.DataFrame(pcs), y], axis = 1)
    y_name = tmp_df.columns[-1]
    targets = targets
    colors = ["r","g", "b", "yellow"]
    if len(y.unique()) > len(colors):
        #colors = sns.color_palette("viridis", 21).as_hex()
        colors = ["#15b01a", "#be0119", "#665fd1", "#ac9362", "#04d9ff", "#98f6b0", "#ff9408", 
                  "#dfc5fe", "#fe46a5", "#056eee", "#0804f9", "#fffd74", "#befd73", "#ec2d01", 
                  "#c4fff7", "#a552e6", "#85a3b2", "#014600", "#4efd54", "#fb5ffc", "#ac7e04"]
    # Plot fig and axes
    fig, axes = plt.subplots(nrows, 3, figsize = figsize, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)
    for target, color in zip(targets,colors):
        i_keep = tmp_df[y_name] == target
        for i ,ax in enumerate(axes.flatten()):
            ax.scatter(tmp_df.loc[i_keep,i], tmp_df.loc[i_keep,i+1], zorder=3, 
                            ec="black", c=color, s=25, alpha = alpha, label = target) 
            ax.set_title(f"PC{i+1} ({var[i]:.2}%) and PC{i+2} ({var[i+1]:.2}%)")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    # Details
    if legend_title is None:
        legend_title = y_name.capitalize()
    legend = ax.legend(title=legend_title, title_fontsize=15,
                        frameon = 1, shadow = True, bbox_to_anchor=legend_anchor)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.suptitle(f"PCA first {n_pcs} PCs{title}", fontsize=16)
    plt.xlabel("First (relative) component", fontsize=13)
    plt.ylabel("Second (relative) component", fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(top=top)
    if save == True:
        plt.savefig(f"{OUT_PATH}{filename}_{FILENAME}.png", dpi = 300)
    plt.show()


# Add wrongly and correctly classified information
def add_confusion_cols(ml_output):
    """
    Add two columns to an input "ml_output".

    One column ("prediction") indicates if a prediction is correct or wrong, 
    and another column ("confusion") including confusion matrix information 
    (TN, TP, FP, FN).

    :ml_output: panda dataframe including yprob, ypred, ytrue (returned by lgb_cv, cnn_cv, or final_pred())
    """  
    ml_output = ml_output.copy()
    # Conditions
    conditions = [
        ((ml_output["ytrue"] == 0) & (ml_output["ypred"] == 0)),
        ((ml_output["ytrue"] == 1) & (ml_output["ypred"] == 1)),
        ((ml_output["ytrue"] == 0) & (ml_output["ypred"] == 1)),
        ((ml_output["ytrue"] == 1) & (ml_output["ypred"] == 0))
        ]
    # List of the values we want to assign for each condition
    values = ['TN', 'TP', 'FP', 'FN']
    # Use np.select to assign values to it using our lists as arguments
    ml_output["confusion"] = np.select(conditions, values)
    # Simply add a column indicating wrong and correct predictions
    ml_output["prediction"] = np.where(ml_output.ytrue == ml_output.ypred, "Correct", "Wrong")
    # If not present add indexes as column
    if any(ml_output.columns.str.contains("index")) == False:
        ml_output = ml_output.reset_index()
    return ml_output


## Threshold-moving ##

# Get optimal binary classification threshold
def get_optimal_threshold(ml_output):
    """
    Compute the threshold-moving (find the optimal threshold for 
    a binary classification problem) using the J-statistics as
    J = Sensitivity + Specificity – 1.

    :ml_output: panda dataframe including yprob, ypred, ytrue (returned by lgb_cv, cnn_cv, or final_pred())
    """  
    # Using J-statistics
    roc_auc, fpr, tpr, thresholds = get_auc_fpr_tpr(ml_output, return_threshold=True)
    J = tpr - fpr              
    ix = np.argmax(J)
    best = thresholds[ix]
    # Output 
    print(f"Optimal threshold = {best:.3}")
    return round(best, 3)

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



###### General hyper-parameter tuning ######

# CV split by chr for par search
def search_cv_split(metadata, n_val_chr=6, max_cv_iter="auto", cv_seed=None):
    """
    Take as input a metadata dataframe and it output a list of tuples where 
    each tuple correspond to a certain fold and it contains two list of 
    indexes, one for the training samples and one for the validation ones.

    It is used to split the data into train and validation during CV for
    parameter optimization. 

    :metadata: pandas dataframe with metadata information

    Optional:
    :n_val_chr: int, number of validation chromosomes used at each fold
    :max_cv_iter: "auto" or int
    :cv_seed: int or None
    """
    val_chr_folds = cv_split(metadata, n_val_chr, cv_seed)
    if max_cv_iter != "auto":
        val_chr_folds = val_chr_folds[0:max_cv_iter]
    print("Validation chromosomes:")
    [print(list(x)) for x in val_chr_folds]
    # Generate a list of tuple where each tuple contain two list of indexes (train and val)
    list_fold_index = []
    for val_chr in val_chr_folds:
        ival = metadata.chr.isin(val_chr)
        list_fold_index.append((list(np.where(~ival)[0]), list(np.where(ival)[0])))
    return list_fold_index



###### LGBM ######

## LGBM scoring functions ##

def lgb_mcc_score(y_pred, y_true):
    """
    Compute MCC as LGBM evaluation function.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """
    y_true = y_true.get_label()
    y_pred = np.round(y_pred)
    is_higher_better = True
    return 'MCC', round(matthews_corrcoef(y_true, y_pred), 5), is_higher_better


def lgb_f1_score(y_pred, y_true):
    """
    Compute F1-score as LGBM evaluation function.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """
    y_true = y_true.get_label()
    y_pred = np.round(y_pred)
    is_higher_better = True
    return 'F1', round(f1_score(y_true, y_pred, average='macro'), 5), is_higher_better


def lgb_precision(y_pred, y_true):
    """
    Compute precision as LGBM evaluation function.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """
    y_true = y_true.get_label()
    y_pred = np.round(y_pred)
    is_higher_better = True
    return 'Precision', round(precision_score(y_true, y_pred), 5), is_higher_better


def lgb_recall(y_pred, y_true):
    """
    Compute recall as LGBM evaluation function.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """

    y_true = y_true.get_label()
    y_pred = np.round(y_pred)
    is_higher_better = True
    return 'Recall', round(recall_score(y_true, y_pred), 5), is_higher_better


def tnr_tpr_score(y_true, y_pred):
    """
    Compute TNR and TPR, used for LGBM evaluation functions.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """

    cf_matrix = confusion_matrix(y_true, y_pred)
    group_counts = [f"{value}" for value in cf_matrix.flatten()]
    row_sums = cf_matrix.sum(axis=1)
    norm_matrix = cf_matrix / row_sums[:, np.newaxis]
    return norm_matrix.flatten()[0], norm_matrix.flatten()[3]


def lgb_tnr_score(y_pred, y_true):
    """
    Compute TNR as LGBM evaluation function.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """

    y_true = y_true.get_label()
    y_pred = np.round(y_pred)
    is_higher_better = True
    return 'TNR', tnr_tpr_score(y_true, y_pred)[0], is_higher_better


def lgb_tpr_score(y_pred, y_true):
    """
    Compute TPR as LGBM evaluation function.
    
    :ytrue: vector of true labels
    :ypred: vector of predicted labels
    """

    y_true = y_true.get_label()
    y_pred = np.round(y_pred)
    is_higher_better = True
    return 'TPR', tnr_tpr_score(y_true, y_pred)[1], is_higher_better

## LGBM plotting functions ##

# Plot metrics after 1 CV iteration
def lgb_plot_eval_result(evals_result, best_iteration, num_iterations, 
                         val_chr=None, name="LGBM"):
    """
    Plot the progressive change of the loss, F1-score, precision, 
    and recall, during the iterations of the LGBM training progress.

    The function is used to evaluate the training progression at the end 
    of each CV iteration.
  
    :evals_result: dictionary used to store all evaluation results by lgb.train() function
    :best_iteration: int, early stopping best iteration
    :num_iterations: int, number of boosting iterations
  
    Optional
    :val_chr: list of string or None
    :name: string
    """

    # Initialize
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2)
    subplot_indexes = (0,0), (0,1), (1,0), (1,1)
    names = list(evals_result["val"].keys())
    for i, indexes in enumerate(subplot_indexes):
        # Plot 
        ax = plt.subplot(gs[indexes])
        lgb.plot_metric(evals_result, metric=names[i], ax=ax)  
        ax.set_title(names[i].replace("_", " "), fontsize = 12)  
        # Add early stopping iteration and legend
        if num_iterations > best_iteration and best_iteration != 0:
            ax.vlines(best_iteration, ymin=-1, ymax=1, color="r", linestyles="dashed", linewidth=0.8, label="best_iter") 
        legend = ax.legend(frameon = 1, shadow = True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black') 
    # Set title and save
    title = "Training progression"
    if val_chr is not None:
        val_chr_str = f"{', '.join(str(i) for i in val_chr)}"
        title += f" ({name}, val on {val_chr_str})"
    else:
        title += f" ({name}, eval on test data)"
    plt.suptitle(title, size=14, y=1.035)
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}{title}.png", dpi = 300, bbox_inches='tight')
    plt.show()


# Plot metrics showing all CV iterations
def plot_loss_lgb_cv(history_lst, val_chrs, percentage_pos, 
                     top=0.93, name="", figsize=(20,18),
                     save=False):      
    """
    Plot the model performance progression during the training progress 
    at each CV iteration.
    
    :history_lst: list of evaluation results ("evals_result" from lgb.train())
    :val_chrs: list of string or None
    :percentage_pos: int or float, percentage of positive samples
    
    Optional
    :top: float or int, the position of the top edge of the subplots
    :name: string
    :figsize: tuple of float or int
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    """

    # Plot    
    colors = sns.color_palette().as_hex()
    nrows = len(history_lst)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=figsize, sharex=True, sharey=False)   
    fig.add_subplot(111, frameon=False)
    
    # Plot each CV iter training progress
    for i, history in enumerate(history_lst):
        # Get the epoch index with the best loss
        no_epochs = np.nanargmin(history["val"]["cross_entropy"]) + 1 
        epochs = list(np.arange(0, no_epochs) + 1) # Don't consider epoch 0
        metrics = list(history["val"].keys())

        # Iterate through the metrics
        for m, metric in enumerate(metrics):
            metric_cap = " ".join([w.capitalize() for w in metric.replace('_', ' ').split()])
            # Train
            axes[i,m].plot(epochs, history["train"][metric][0:no_epochs], 
                        label= f"Train {metric_cap}", zorder=3)
            # Val
            axes[i,m].plot(epochs, history["val"][metric][0:no_epochs], 
                        label= f"Val {metric_cap}", zorder=3)
            # Details
            axes[i,m].set_title(f"{metric_cap} {i+1}° fold\nVal = {val_chrs[i]} ({int(percentage_pos[i]*100)}% pos)")
            axes[i,m].set_ylabel(metric_cap)
            axes[i,m].grid(zorder=0, color="lightgray")
            axes[i,m].grid(zorder=0, color="lightgray")
            legend = axes[i,m].legend(frameon = 1, shadow = True, fontsize=11)
            legend = axes[i,m].legend(frameon = 1, shadow = True, fontsize=11)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')   
        
    # General details
    plt.tick_params(labelcolor='none', top=False, bottom=True, left=False, right=False)
    plt.xlabel("Epoch")
    fig.suptitle(f"CV LGBM training progression", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=top)
    if save:
        plt.savefig(f"{OUT_PATH}lgb_loss_cv_progression_{name}_{FILENAME}.png", 
                    dpi = 300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()


## Other LGBM functions ##

def lgb_class_weights(ytrain, class_weight, par):
    """
    Compute LGBM class weights as suggested by TensorFlow 
    tutorial for unbalanced classification problem:
    Negative weight = (1 / number of negative samples) * total number of samples / 2
    Positive weight = (1 / number of positive samples) * total number of samples / 2

    :ytrain: vector of train samples true labels
    :class_weight: string ("auto" or "balanced") or None, if "auto" it uses the custom weights,
    if "balanced" LGBM computes the weights internally
    :par: dictionary of LGBM parameters
    """

    if class_weight == "auto":
        neg_train, pos_train = np.bincount(ytrain)
        total_train = neg_train + pos_train
        weight_for_0 = (1 / neg_train)*(total_train)/2.0 
        weight_for_1 = (1 / pos_train)*(total_train)/2.0
        par["scale_pos_weight"] = round(weight_for_1 / weight_for_0, 2)
    elif class_weight == "balanced":
        par["is_unbalance "] = True
    return par


def lgb_print_par(par, num_iterations, early_stopping_rounds):
    """
    Print LGBM parameters.

    :par: dictionary of LGBM parameters
    :num_iterations: int, number of boosting iterations
    :early_stopping_rounds: int or None, number of iterations before early stopping
    """

    print("\nParameters:")
    print(f"  num_iterations = {num_iterations}")
    print(f"  early_stopping_rounds = {early_stopping_rounds}")
    [print(f"  {x} = {par[x]}") for x in par]


## LGBM training functions ##

def lgb_pred(model, x):
    """
    Predict the probability of class 1 and the predicted label.

    :model: LGBM model
    :x: dataframe including the samples to predict
    """
    y_prob = model.predict(np.array(x))
    y_pred = np.round(y_prob)
    return y_prob, y_pred
  

def lgb_train_pred(xtrain, ytrain, xtest, ytest, par, 
                   num_iterations=1500, early_stopping_rounds=None, 
                   metrics="loss", val_chr=None, 
                   verbose=50, name="LGBM"):
    """
    The function train a LGBM model on the training data and it 
    evaluate it on the validation/test data. 

    If "verbose" > 0, it print the model performance on train and 
    validation/test data. If "verbose" > 0 and "metrics" is set
    to "all" (it slow down the training progress), it also plot 
    the progressive change of the model performance (loss, F1-score, 
    precision, and recall) during the iterations of the training progress. 

    :xtrain: dataframe, training data
    :ytrain: vector of training samples true labels
    :xtest: dataframe, test data 
    :ytest: vector of test samples true labels
    :par: dictionary of LGBM parameters

    Optional
    :num_iterations: int, number of boosting iterations
    :early_stopping_rounds: int or None, number of iterations before early stopping
    :metrics: string, evaluation functions ("default" : loss on val, 
    "all": loss, f1, recall, precision on train and val, 
    "f1_score": loss and f1 on val, "loss" : loss on train and val)
    :val_chr: list of string or None
    :verbose: int 
    :name: string
    """

    start_time = datetime.now()
    # Train
    train_data = lgb.Dataset(xtrain , label = ytrain, 
                            free_raw_data = False)
    valid_data = lgb.Dataset(xtest, label = ytest,
                            free_raw_data = False)
    evals_result = {}
    # Define the model
    if metrics != "default":
        if metrics == "all":
            model = lgb.train(par, train_data, valid_sets=[train_data, valid_data], valid_names=["train", "val"], 
                              feval = lambda y_pred, y_true: [lgb_f1_score(y_pred, y_true), 
                                                              lgb_recall(y_pred, y_true),
                                                              lgb_precision(y_pred, y_true)],      
                              verbose_eval=verbose, evals_result=evals_result, num_boost_round=num_iterations, 
                              early_stopping_rounds=early_stopping_rounds)
        elif metrics == "f1_score":  
            model = lgb.train(par, train_data, valid_sets=[valid_data], valid_names=["val"], 
                            feval = lambda y_pred, y_true: lgb_f1_score(y_pred, y_true), 
                            verbose_eval=verbose, evals_result=evals_result, num_boost_round=num_iterations, 
                            early_stopping_rounds=early_stopping_rounds)
        elif metrics == "loss":  
            model = lgb.train(par, train_data, valid_sets=[train_data, valid_data], valid_names=["train", "val"], 
                              verbose_eval=verbose, num_boost_round=num_iterations, 
                              early_stopping_rounds=early_stopping_rounds, evals_result=evals_result)
        else:
            raise NameError("metrics argument must include one of the following: \"default\", \"all\", \"f1_score\", \"loss\", \"loss_mcc\"")
    else:
        model = lgb.train(par, train_data, valid_sets=[valid_data], valid_names=["val"], 
                          verbose_eval=verbose, num_boost_round=num_iterations, 
                          early_stopping_rounds=early_stopping_rounds)
    # Predict
    ytrain_prob, ytrain_pred = lgb_pred(model, xtrain)
    ytest_prob, ytest_pred = lgb_pred(model, xtest)
    if verbose > 0:
        # Evaluate
        print("")
        print_eval(ytrain, ytrain_pred, ytest, ytest_pred)
        print("")
        if metrics == "all":
            lgb_plot_eval_result(evals_result, model.best_iteration, num_iterations, val_chr, name)
        # Report time 
        print(f"Duration: {datetime.now() - start_time}")
    return model, ytrain_prob, ytrain_pred, ytest_prob, ytest_pred, evals_result


def lgb_cv(xtrain, ytrain, metadata, par, 
           num_iterations=1500, early_stopping_rounds=None,
           n_val_chr=3, cv_seed=None, normalization=True, 
           class_weight=None, metrics="loss", 
           final_cv_eval=True, verbose=50, name="LGBM",
           plt_figsize=(20,18), plt_top=0.93, plt_save=False):
    """
    Final function used to train LGBM using CV. 
    
    At each CV iteration, the train data is split into train and validation 
    and a StandardScaler is fitted to the training data, it is saved 
    (MODEL_PATH directory as .pk1) and it is applied to both training 
    and validation data. Then, a LGBM model is trained, saved 
    (MODEL_PATH as .txt) and evaluated. If "metrics" is set to "all" 
    (it slow down the training progress), it generate an evaluation plot 
    showing the training progression at the end of each CV iteration and a 
    final evaluation plot showing the training progression on all CV 
    iterations. The function output three pandas dataframes, two including 
    the results obtained on the training and validation data and one with 
    the model feature importance computed during CV.

    :xtrain: dataframe, training data
    :ytrain: vector of training samples true labels
    :metadata: pandas dataframe with metadata information
    :par: dictionary of LGBM parameters

    Optional
    :num_iterations: int, number of boosting iterations
    :early_stopping_rounds: int or None, number of iterations before early stopping
    :n_val_chr: int, number of validation chromosomes used at each iteration
    :cv_seed: int
    :normalization: boolean
    :class_weight: string ("auto" or "balanced") or None, if "auto" it uses the 
    custom weights, if "balanced" LGBM computes the weights internally
    :metrics: string, evaluation functions ("default" : loss on val, 
    "all" : loss, f1, recall, precision on train and val, 
    "f1_score" : loss and f1 on val, "loss" : loss on train and val)
    :final_cv_eval: boolean
    :verbose: int
    :name: string
    :plt_figsize: tuple of int or float
    :plt_top: float or int, the position of the top edge of the subplots
    :plt_save: boolean, if True it save the final plot in OUT_PATH directory as .png
    """

    # Initialize time
    start_time = datetime.now()
    if verbose > 0:
        print(f"Performing {name} CV")
    
    # Initialize 1d df to store predictions and true values
    train_yprob_vec, train_ypred_vec, train_ytrue_vec = np.array([]), np.array([]), np.array([])
    val_yprob_vec, val_ypred_vec, val_ytrue_vec = np.array([]), np.array([]), np.array([])
    importance = pd.DataFrame()
    importance["feature"] = list(xtrain.columns)
    histories = []
    best_iter_lst = []
    val_chr_str_lst = []

    # Print parameters
    if verbose > 0:
        lgb_print_par(par, num_iterations, early_stopping_rounds)

    # CV by chromosomes
    val_chr_lst = []
    percentage_pos_lst = []
    val_chr_folds = cv_split(metadata, n_val_chr, cv_seed)

    # Split according to the fold
    for i, val_chr in enumerate(val_chr_folds):
        val_chrs = ', '.join(str(i) for i in val_chr)
        val_chr_lst.append(val_chrs)
        if verbose > 0:
            print(f"\n> Starting CV iteration {i+1}, valid chrs = {val_chrs}") 
        ival = metadata.chr.isin(val_chr)
        X_train, X_val = xtrain[~ival], xtrain[ival]
        y_train, y_val = ytrain[~ival], ytrain[ival]

        # Get percentage positive samples
        total_train, neg_train, pos_train, total_val, neg_val, pos_val = get_no_samples(y_train, y_val, verbose=verbose, return_train_only=False)
        percentage_pos_lst.append(round(pos_val/total_val, 2))

        # Set class weight
        if class_weight is not None:
            par = lgb_class_weights(y_train, class_weight, par)
            print("Positive weight =", par["scale_pos_weight"])

        # Normalization
        if normalization:
            X_train, X_val = normalize(X_train, X_val, name=f"{name}_cv_{i+1}", save=True)

        # Train and evaluate
        model, train_yprob, train_ypred, val_yprob, val_ypred, evals_result = lgb_train_pred(X_train, y_train, X_val, y_val, par=par,
                                                                                            num_iterations=num_iterations,
                                                                                            early_stopping_rounds=early_stopping_rounds,
                                                                                            metrics=metrics, val_chr=val_chr, 
                                                                                            verbose=verbose, name=f"{name}_cv_{i+1}")
        importance[f"iteration_{i+1}"] = model.feature_importance(importance_type='gain')

        # Store prediction on training data
        train_yprob_vec = np.concatenate((train_yprob_vec, train_yprob))
        train_ypred_vec = np.concatenate((train_ypred_vec, train_ypred))
        train_ytrue_vec = np.concatenate((train_ytrue_vec, y_train))
        # Store prediction on validation data
        val_yprob_vec = np.concatenate((val_yprob_vec, val_yprob))
        val_ypred_vec = np.concatenate((val_ypred_vec, val_ypred))
        val_ytrue_vec = np.concatenate((val_ytrue_vec, y_val))
        # Store history
        histories.append(evals_result)   
        best_iter_lst.append(model.best_iteration)
        # Save model
        filepath = f"{MODEL_PATH}{name}_cv_{i+1}_model_{FILENAME}.txt"
        print(f"Saving model from best iteration:\n\t{filepath}")
        model.save_model(filepath, num_iteration = model.best_iteration)
    
    # Stack training and validation predictions into two panda df
    train_output = pd.DataFrame({"yprob": train_yprob_vec, "ypred": train_ypred_vec,"ytrue": train_ytrue_vec})
    val_output = pd.DataFrame({"yprob": val_yprob_vec, "ypred": val_ypred_vec,"ytrue": val_ytrue_vec})
    importance["average"] = np.mean(importance, 1)
    # Plot the loss progression in each CV iter
    if metrics == "all":
        plot_loss_lgb_cv(histories, val_chr_lst, percentage_pos_lst, 
                        name=name, save=plt_save,
                        figsize=plt_figsize, top=plt_top)
    # Final report
    if final_cv_eval:
        print(f"\n>> {name} CV final report")
        print_output_eval(train_output, val_output)
        # Report time 
        print(f"Duration: {datetime.now() - start_time}")
    return train_output, val_output, importance  #, (histories, val_chr_lst, percentage_pos_lst)


### LGBM parameter tuning ###

## Functions used in Sci-kit optimize and HyperOpt ##

def lgb_train_pred_opt(xtrain, ytrain, xtest, ytest, par, 
                       num_iterations=1500, early_stopping_rounds=None, 
                       val_chr=None):
    """
    Simple training and evaluation function used to perform LGBM
    parameter optimization using Sci-kit optimize and HyperOpt.

    :xtrain: dataframe, training data
    :ytrain: vector of training samples true labels
    :xtest: dataframe, validation or test data
    :ytest: vector of validation or test samples true labels
    :par: dictionary of LGBM parameters

    Optional
    :num_iterations: int, number of boosting iterations
    :early_stopping_rounds: int or None, number of iterations before early stopping
    :val_chr: list of string or None
    """

    # Train
    train_data = lgb.Dataset(xtrain , label = ytrain, free_raw_data = False)
    valid_data = lgb.Dataset(xtest, label = ytest, free_raw_data = False)
    model = lgb.train(par, train_data, valid_sets=[valid_data], valid_names=["val"], 
                      feval= lambda y_pred, y_true: lgb_f1_score(y_pred, y_true),
                      num_boost_round=num_iterations, verbose_eval=0,
                      early_stopping_rounds=early_stopping_rounds)
    # Predict
    ytrain_prob, ytrain_pred = lgb_pred(model, xtrain)
    ytest_prob, ytest_pred = lgb_pred(model, xtest)
    return ytest_pred


def lgb_cv_opt(xtrain, ytrain, metadata, par,
               num_iterations=1500, early_stopping_rounds=None,
               n_val_chr=3, cv_seed=None, normalization=True, 
               print_eval=False):
    """
    Training and evaluation CV function used to perform LGBM
    parameter optimization using Sci-kit optimize and HyperOpt.

    :xtrain: dataframe, training data
    :ytrain: vector of training samples true labels
    :metadata: pandas dataframe with metadata information
    :par: dictionary of LGBM parameters

    Optional
    :num_iterations: int, number of boosting iterations
    :early_stopping_rounds: int or None, number of iterations before early stopping
    :n_val_chr: int, number of validation chromosomes used at each iteration
    :cv_seed: int
    :normalization: boolean
    :print_eval: boolean
    """

    start_time = datetime.now()
    # Initialize a list to store scores
    f1_lst = []
    tnr_lst = []
    tpr_lst = []
    # CV by chromosomes
    val_chr_folds = cv_split(metadata, n_val_chr, cv_seed)
    for i, val_chr in enumerate(val_chr_folds):
        ival = metadata.chr.isin(val_chr)
        X_train, X_val = xtrain[~ival], xtrain[ival]
        y_train, y_val = ytrain[~ival], ytrain[ival]
        # Normalization
        if normalization:
            X_train, X_val = normalize(X_train, X_val)
        # Train, predict and store score
        val_ypred = lgb_train_pred_opt(X_train, y_train, X_val, y_val, par=par,
                                       num_iterations=num_iterations,
                                       early_stopping_rounds=early_stopping_rounds,
                                       val_chr=val_chr)
        f1_lst.append(f1_score(y_val, val_ypred, average='macro'))    
        tnr, tpr = get_tnr_tpr(y_val, val_ypred, digit=4)
        tnr_lst.append(tnr)
        tpr_lst.append(tpr)    

    # Report time, score, and parameters
    if print_eval:
        avg_f1, avg_tnr, avg_tpr = np.mean(f1_lst), np.mean(tnr_lst), np.mean(tpr_lst)
        print(f"Duration: {datetime.now() - start_time} | F1-score: {avg_f1} | TNR: {avg_tnr} | TPR: {avg_tpr}")
    # Convert from a maximizing score to a minimizing score
    return 1.0 - avg_f1


# LGBM tuning with HyperOpt
def lgb_hyperopt(x_train, y_train, metadata_train, 
                 fixed_params, pbounds, opt_iteratations=50, 
                 num_iterations=1500, early_stopping_rounds=150, 
                 n_val_chr=3, return_best_only=False, opt_seed=1, 
                 print_eval=False, cv_seed=33):
    """  
    The function uses BayesianOptimization package to perform LGBM 
    Bayesian hyperparameters tuning by optimazing the average 
    F1-score obtained during the CV iterations. 
    
    To optimize a different set of parameters, these must be changed 
    in the "lgb_fun" internal function.

    :x_train: dataframe, training data
    :y_train: vector of training samples true labels
    :metadata_train: pandas dataframe with metadata information
    :params: dictionary of LGBM parameters 
    :pbounds: dictionary of bounds for the parameters to optimize

    Optional
    :num_iterations: int, number of boosting iterations
    :opt_init_points: int, number of initialization iteration before starting the search
    :opt_iteratations: int, number of optimization iteration
    :early_stopping_rounds: int or None, number of iterations before early stopping
    :n_val_chr: int, number of validation chromosomes used at each iteration
    :opt_seed: int
    :cv_seed: int
    :opt_verbose: int
    :cv_verbose: int
    :final_cv_eval: boolean
    """

    def objective_function(tuning_params):
        """
        Output the average F1-score across the CV iterations.

        Internal function that takes as argument the parameters to optimize, 
        and it uses lgb_cv function to train and evaluate the model using CV.
        """    
        print(tuning_params)
        for p in tuning_params:
            fixed_params[p] = tuning_params[p]
            neg_avg_f1 = lgb_cv_opt(X_TRAIN, y_TRAIN, metadata_train, fixed_params, 
                                    num_iterations=num_iterations, 
                                    early_stopping_rounds=early_stopping_rounds,
                                    n_val_chr=n_val_chr, cv_seed=cv_seed, print_eval=print_eval)
        return {'loss': neg_avg_f1, 'status': STATUS_OK}   
  
    # Perform optimization
    trials = Trials()
    best_param = fmin(objective_function, pbounds, algo=tpe.suggest, max_evals=opt_iteratations, 
                      trials=trials, rstate= np.random.RandomState(1))
    # Output
    if return_best_only:
        return best_param
    else:
        return best_param, trials


# BayesianOptimization with parameters to prevent overfitting
def lgb_bayes_opt(x_train, y_train, metadata_train, 
                  params, pbounds, num_iterations=1500, 
                  opt_init_points=5, opt_iteratations=50,
                  early_stopping_rounds=150, 
                  n_val_chr=3, opt_seed=1, cv_seed=33, 
                  opt_verbose=2, cv_verbose=0, final_cv_eval=False):      
    """  
    The function uses BayesianOptimization package to perform LGBM 
    Bayesian hyperparameters tuning by optimazing the average 
    F1-score obtained during the CV iterations. 
    
    To optimize a different set of parameters, these must be changed 
    in the "lgb_fun" internal function.

    :x_train: dataframe, training data
    :y_train: vector of training samples true labels
    :metadata_train: pandas dataframe with metadata information
    :params: dictionary of LGBM parameters 
    :pbounds: dictionary of bounds for the parameters to optimize

    Optional
    :num_iterations: int, number of boosting iterations
    :opt_init_points: int, number of initialization iteration before starting the search
    :opt_iteratations: int, number of optimization iteration
    :early_stopping_rounds: int or None, number of iterations before early stopping
    :n_val_chr: int, number of validation chromosomes used at each iteration
    :opt_seed: int
    :cv_seed: int
    :opt_verbose: int
    :cv_verbose: int
    :final_cv_eval: boolean
    """

    def lgb_fun(learning_rate, num_leaves, max_depth, lambda_l1, lambda_l2, 
                feature_fraction, bagging_fraction, min_data_in_leaf, 
                min_data_in_bin, min_gain_to_split, min_child_weight):
        """
        It output the average F1-score across the CV iterations.

        Internal function that takes as argument the parameters to optimize, 
        and it uses lgb_cv function to train and evaluate the model using CV.
        """   

        # Assign tuning parameters
        params["learning_rate"] = learning_rate
        #params["scale_pos_weight"] = scale_pos_weight
        params["num_leaves"] = round(num_leaves)
        params["max_depth"] = round(max_depth)
        params['lambda_l1'] = max(0, lambda_l1)
        params['lambda_l2'] = max(0, lambda_l2)
        params["feature_fraction"] = feature_fraction
        params["bagging_fraction"] = feature_fraction
        params["min_data_in_leaf"] = round(min_data_in_leaf)
        params["min_data_in_bin"] = round(min_data_in_bin)
        params["min_gain_to_split"] = min_gain_to_split
        if min_child_weight > 1: 
            params["min_child_weight"] = round(min_child_weight)
        else:
            params["min_child_weight"] = min_child_weight
        # LGBM CV
        _, cv_results, _ = lgb_cv(x_train, y_train, metadata_train, params, 
                                  num_iterations=num_iterations, 
                                  early_stopping_rounds=early_stopping_rounds,
                                  metrics="f1_score", n_val_chr=n_val_chr, 
                                  cv_seed=cv_seed, verbose=cv_verbose, 
                                  final_cv_eval=final_cv_eval)
        avg_f1 = f1_score(cv_results["ytrue"], cv_results["ypred"], average='macro')                                                 
        return avg_f1

    # Perform optimization
    optimizer = BayesianOptimization(f=lgb_fun, pbounds=pbounds,
                                     verbose=opt_verbose, # verbose = 1 prints only when max is observed, verbose = 0 is silent
                                     random_state=opt_seed)
    optimizer.maximize(init_points=opt_init_points, n_iter=opt_iteratations)
    return optimizer



###### CNN ######

# Callbacks for accurate monitoring of loss and metrics   
class Metrics(Callback):
    """
    Custom callback used to monitor the real model performance (loss, 
    F1-score, precision, and recall) on training and validation data.
    """
    
    def __init__(self, monitoring_data):   
        """
        Initialize the input data.
        
        :monitoring_data: tuple (xtrain, ytrain, xtest, ytest)
        """

        # Monitoring data must be a tuple as ()
        self.monitoring_data = monitoring_data    
        
    def on_train_begin(self, logs={}):  
        """
        Action to perform before starting the training progress.
        Initialize variables, assess model performance before
        starting the training.
        """

        # Initialize lists
        self.train_losses, self.train_f1s, self.train_recalls, self.train_precisions = [], [], [], []
        self.val_losses, self.val_f1s, self.val_recalls, self.val_precisions = [], [], [], []
        
        # Assign true labels
        self.train_targ = self.monitoring_data[1] 
        self.val_targ = self.monitoring_data[3]   
        
        # Predicted labels, and their prob before training
        train_predict_prob_0 = (np.asarray(self.model.predict(self.monitoring_data[0])))
        train_predict_0 = train_predict_prob_0.round()
        val_predict_prob_0 = (np.asarray(self.model.predict(self.monitoring_data[2])))
        val_predict_0 = val_predict_prob_0.round()
        
        # Compute metrics at epoch 0 (before starting training) 
        train_loss_0 = round(log_loss(self.train_targ, train_predict_prob_0), 5)
        val_loss_0 = round(log_loss(self.val_targ, val_predict_prob_0), 5) 
        train_f1_0 = round(f1_score(self.train_targ, train_predict_0, average='macro'), 5)
        val_f1_0 = round(f1_score(self.val_targ, val_predict_0, average='macro'), 5)
        train_precision_0 = round(precision_score(self.train_targ, train_predict_0), 5)
        val_precision_0 = round(precision_score(self.val_targ, val_predict_0), 5)
        train_recall_0 = round(recall_score(self.train_targ, train_predict_0), 5)
        val_recall_0 = round(recall_score(self.val_targ, val_predict_0), 5)
        
        # Store results
        self.train_losses.append(train_loss_0)
        self.train_f1s.append(train_f1_0)
        self.train_recalls.append(train_recall_0)
        self.train_precisions.append(train_precision_0)
        self.val_losses.append(val_loss_0)
        self.val_f1s.append(val_f1_0)
        self.val_recalls.append(val_recall_0)
        self.val_precisions.append(val_precision_0)
        print(f' - Loss_0: {train_loss_0} - F1_0: {train_f1_0} - Train_Precision_0: {train_precision_0}, - Train_Recall_0: {train_recall_0}')
        print(f' - Val_Loss_0: {val_loss_0} - Val_F1_0: {val_f1_0} - Val_Precision_0: {val_precision_0}, - Val_Recall_0: {val_recall_0}')
        
    def on_epoch_end(self, epoch, logs={}):
        """
        Action to perform at the end of each epoch.
        Assess model performance at the end of each epoch.
        """

        # Get true, predicted labels, and their prob
        train_predict_prob = (np.asarray(self.model.predict(self.monitoring_data[0])))
        train_predict = train_predict_prob.round()  
        val_predict_prob = (np.asarray(self.model.predict(self.monitoring_data[2])))
        val_predict = val_predict_prob.round()
        
        # Compute loss and metrics
        train_loss = round(log_loss(self.train_targ, train_predict_prob), 5)
        val_loss = round(log_loss(self.val_targ, val_predict_prob), 5)
        train_f1 = round(f1_score(self.train_targ, train_predict, average='macro'), 5)
        val_f1 = round(f1_score(self.val_targ, val_predict, average='macro'), 5)
        train_precision = round(precision_score(self.train_targ, train_predict), 5)
        val_precision = round(precision_score(self.val_targ, val_predict), 5)
        train_recall = round(recall_score(self.train_targ, train_predict), 5)
        val_recall = round(recall_score(self.val_targ, val_predict), 5)
        
        # Store results
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_f1s.append(train_f1)
        self.val_f1s.append(val_f1)
        self.train_recalls.append(train_recall)
        self.val_recalls.append(val_recall)
        self.train_precisions.append(train_precision)
        self.val_precisions.append(val_precision)
        print(f' - Loss: {train_loss} - F1: {train_f1} - Precision: {train_precision}, - Recall: {train_recall} - Val_loss: {val_loss} - Val_F1: {val_f1} - Val_Precision: {val_precision}, - Val_Recall: {val_recall}')



# Define the model
def cnn_model(name=name, lr=0.0001, len_win=301, n_features=1, summary=False, print_name=True):
    """
    Define a CNN model. 
    
    To train a different CNN model, this function must be overwritten (defining a 
    different architecture ) before running the CNN training function cnn_cv().

    Otional
    :name: string
    :lr: int or float, learning rate
    :len_win: int, length of the genomic profiles
    :n_features: int, 1 for 1D CNN or 2 of 2D CNN
    :summary: boolean
    :print_name: boolean
    """  

    if print_name:
        print("Model:", name)
    model = Sequential(name=name)

    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", 
                     input_shape=(len_win, n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5))

    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5))

    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5, seed=CNN_SEED))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), 
                  loss='binary_crossentropy', metrics=METRICS)
    if summary:
        model.summary()
    return model


# Oversample the positive class
def upsample(xtrain, ytrain):
    """
    Upsampling function for class imbalances (upsampling of the positive class).

    :xtrain: dataframe, training data
    :xtest: dataframe, test data 
    """  
    
    # Get positive and negative samples
    pos_xtrain = xtrain[ytrain==1]
    neg_xtrain = xtrain[ytrain==0]
    pos_ytrain = ytrain[ytrain==1]
    neg_ytrain = ytrain[ytrain==0]

    # Upsampling
    ids = np.arange(len(pos_xtrain))
    choices = np.random.choice(ids, len(neg_xtrain))
    res_pos_xtrain = pos_xtrain[choices,:]
    res_pos_ytrain = pos_ytrain.iloc[choices]

    # Merge negative and positive samples
    resampled_xtrain = np.concatenate([res_pos_xtrain, neg_xtrain], axis=0)
    resampled_ytrain = np.concatenate([res_pos_ytrain, neg_ytrain], axis=0)

    # Shuffle
    order = np.arange(len(resampled_ytrain))
    np.random.shuffle(order)
    resampled_xtrain = resampled_xtrain[order]
    resampled_ytrain = resampled_ytrain[order]
    return resampled_xtrain, resampled_ytrain

## CNN Plotting functions ##

#  Plot CNN evaluation during training (after 1 CV iteration)
def plot_cnn_eval(train_history):
    """
    Function used to plot the CNN training progression (loss, 
    ROC-AUC, precision, rcall) at the end of each CV iteration.

    :train_history: history resulting from keras model.fit()
    """  
    # Initialize
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6), sharex=True, sharey=False)
    names = "loss", "auc", "precision", "recall"
    for i ,ax in enumerate(axes.flatten()):
        # Plot AUC
        if i == 1:
            ax.plot(train_history.history[names[i]], label=names[i].upper(), zorder=3)
            ax.plot(train_history.history[f'val_{names[i]}'], label = f'Val {names[i].upper()}', zorder=3)
            ax.set_title(names[i].upper())
            ax.set_ylabel(names[i].upper())
            ax.set_ylim([0.5, 1])
        # Other plots
        else:
            ax.plot(train_history.history[names[i]], label=names[i].capitalize(), zorder=3)
            ax.plot(train_history.history[f'val_{names[i]}'], label = f'Val {names[i].capitalize()}', zorder=3)
            ax.set_title(names[i].capitalize())
            ax.set_ylabel(names[i].capitalize())
            if i > 1:
                ax.set_ylim([0, 1])
        ax.grid(zorder=0, color="lightgray")
        ax.legend()

    fig.tight_layout()
    plt.show()


# Plot CNN evaluation during training (after all CV iterations)
def plot_loss_cnn_cv(history_object, val_chrs, percentage_pos, top=1.2, name="", figsize=(6.5,22),
                     legend_anchor=(3.5, 0.5), save=False):  
    """   
    Plot the real model performance progression (loss, F1-score, precision, 
    and recall) during the training progress at each CV iteration.

    It takes as input the result of Metrics() class used as Keras custom callback.
    
    :history_object: Keras custom callback (Metrics() class)
    :val_chrs: list of string or None
    :percentage_pos: int or float, percentage of positive samples
    
    Optional
    :top: float or int, the position of the top edge of the subplots
    :name: string
    :figsize: tuple of float or int
    :legend_anchor: tuple of float or int, legend box position
    :save: boolean, if True it save the plot in OUT_PATH directory as .png
    """
    
    # Plot
    n_metrics = 4  
    metrics = ["Loss", "F1", "Recall", "Precision"]
    colors = sns.color_palette().as_hex()
    fig, axes = plt.subplots(nrows=len(history_object), ncols=n_metrics, figsize=figsize, sharex=True, sharey=False)

    fig.add_subplot(111, frameon=False)

    # Plot each CV iter training progress
    for i, i_cv_history in enumerate(history_object):
        
        # Get the history object as a dictionary and its keys
        history_dict = vars(i_cv_history)
        vars_lst = [k for k in vars(i_cv_history).keys()]
        vars_lst = vars_lst[3:11]

        # Get the epoch index with the best val loss
        no_epochs = np.nanargmin(history_dict[vars_lst[4]]) + 1  
        epochs = list(range(0, no_epochs))
        ###print(f"CV {i+1}, Best epoch {no_epochs-1}")
      
        # Iterate through the metrics
        for m in range(n_metrics):
            
            # Assign train variables of history_object
            train_var = vars_lst[m]
            # Assign val variables of history_object
            val_var = vars_lst[m+4]
        
            # Train
            axes[i,m].plot(epochs, history_dict[train_var][0:no_epochs], 
                           label= train_var, zorder=3)
            # Val
            axes[i,m].plot(epochs, history_dict[val_var][0:no_epochs], 
                           label= val_var, zorder=3)
            # Details fold
            axes[i,m].set_title(f"{metrics[m]} {i+1}° fold\nVal = {val_chrs[i]} ({int(percentage_pos[i]*100)}% pos)")
            axes[i,m].set_ylabel(metrics[m])
            axes[i,m].grid(zorder=0, color="lightgray")
            legend = axes[i,m].legend(frameon = 1, shadow = True, fontsize=11)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')    
    # General details 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Epoch")
    fig.suptitle(f"CV CNN training progression", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=top)
    if save:
        plt.savefig(f"{OUT_PATH}cnn_loss_cv_progression_{name}_{FILENAME}.png", 
                    dpi = 300, bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()


### CNN training functions ###

def cnn_train_pred(xtrain, ytrain, xtest, ytest, model, 
                   class_weight=None, epochs=50, batch_size=128, norm=True,
                   upsampling=False, plot_eval=True, early_stopping=None,
                   reshape_to_tensor=True, n_features=1, custom_callbacks=False,
                   verbose=1, name=""):
    """
    The function train a CNN model on the training data and it evaluate it on 
    the validation/test data. 

    If "custom_callbacks" is set to True (it slow down the training progress), 
    it also monitor the real model performance during training progression.

    :xtrain: dataframe, training data
    :ytrain: vector of training samples true labels
    :xtest: dataframe, test data 
    :ytest: vector of test samples true labels
    :model: Keras model

    Optional
    :class_weight: None, "auto" or dictionary inclyding binary class weigths
    :epochs: int
    :batch_size: int
    :norm: boolean, if True fit, apply and save a StandardScaler
    :upsampling: boolean, if True apply upsampling on the positive class
    :plot_eval: boolean, if True plot training progression
    :early_stopping: None or Keras "early_stopping" callback
    :reshape_to_tensor: boolean, if True reshape the input to tensor
    :n_features: int, 1 for 1D CNN and 2 for 2D CNN
    :custom_callbacks: boolean, if True it use Metrics() class to monitor training
    :verbose: int
    :name: string
    """

    start_time = datetime.now()

    # Normalization
    if norm:
        xtrain, xtest = normalize(xtrain, xtest, save=True, name=name)
    
    # Rehape
    if reshape_to_tensor == True:
        xtrain = xtrain.values.reshape((xtrain.shape[0], -1, n_features))
        xtest = xtest.values.reshape((xtest.shape[0], -1, n_features))
    
    # Upsampling training set
    if upsampling:
        xtrain, ytrain = upsample(xtrain, ytrain)
    
    # Compute class weight
    total_train, neg_train, pos_train = get_no_samples(ytrain, ytest, verbose=verbose)
    if class_weight == "auto":
        weight_for_0 = (1 / neg_train)*(total_train)/2.0 
        weight_for_1 = (1 / pos_train)*(total_train)/2.0
        class_weight = {0: round(weight_for_0, 2), 1: round(weight_for_1, 2)}
    print(f"Class weights = {class_weight}")
  
    # Initialize callbacks 
    filepath = f"{MODEL_PATH}{name}_model_{FILENAME}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks = [checkpoint]   
    if custom_callbacks:
        custom_metrics = Metrics(monitoring_data=(xtrain, ytrain, xtest, ytest))
        callbacks.append(custom_metrics)
    if early_stopping is not None:
        callbacks.append(early_stopping)
    print("Callbacks:")
    [print("\t", cb) for cb in callbacks]  
    
    # Train
    history = model.fit(xtrain, ytrain, epochs=epochs, validation_data=(xtest, ytest), 
                        batch_size=batch_size, class_weight=class_weight, verbose=verbose,
                        callbacks=callbacks, shuffle=True)  
    print(f"Saving and loading model from best epoch:\n\t{filepath}")
    model = load_model(filepath)  

    # Pred
    train_prob = model.predict(xtrain).reshape(xtrain.shape[0])
    train_pred = np.round(train_prob)
    test_prob = model.predict(xtest).reshape(xtest.shape[0])
    test_pred = np.round(test_prob)

    # Store pred
    train_output = pd.DataFrame({"yprob": train_prob, "ypred": train_pred,"ytrue": ytrain})
    test_output = pd.DataFrame({"yprob": test_prob, "ypred": test_pred,"ytrue": ytest})
    
    # Eval
    if plot_eval == True:
        print("")
        plot_cnn_eval(history)
        print("")
    print_eval(ytrain, train_pred, ytest, test_pred)

    # Report time 
    print(f"Duration: {datetime.now() - start_time}")
    if custom_callbacks:
        return train_output, test_output, history, custom_metrics
    else:
        return train_output, test_output, history


def cnn_cv(xtrain, ytrain, metadata,
           early_stopping=None, n_val_chr=3, cv_seed=None, custom_callbacks=False,
           verbose=0, name="CNN", class_weight=None, epochs=200, batch_size=2048, 
           norm=True, upsampling=False, reshape_to_tensor=True, n_features=1, plot_eval=True,
           plt_figsize=(20,18), plt_top=0.93, plt_save=False):
    """
    Final function used to train a CNN using CV. 
    
    At each CV iteration, the train data is split into train and validation 
    and a StandardScaler is fitted to the training data, it is saved 
    (MODEL_PATH directory as .pk1) and it is applied to both training and 
    validation data. Then, a CNN model is trained, saved (MODEL_PATH as .h5) 
    and evaluated. To use a different model the cnn_model() function must be 
    overwritten. If "custom_callbacks" is set to True (it slow down the 
    training progress), it also monitor the real model performance during 
    training progression. The function output two pandas dataframes including 
    the results obtained on the training and validation data.

    :xtrain: dataframe, training data
    :ytrain: vector of training samples true labels
    :metadata: pandas dataframe with metadata information

    Optional
    :early_stopping: None or Keras "early_stopping" callback
    :n_val_chr: int, number of validation chromosomes used at each iteration
    :cv_seed: int
    :custom_callbacks: boolean, if True it use Metrics() class to monitor training
    :verbose: int
    :name: string
    :class_weight: None, "auto" or dictionary inclyding binary class weigths
    :epochs: int
    :batch_size: int
    :norm: boolean, if True fit, apply, and save a StandardScaler
    :upsampling: boolean, if True apply upsampling on the positive class
    :reshape_to_tensor: boolean, if True reshape the input to tensor
    :n_features: int, 1 for 1D CNN and 2 for 2D CNN
    :plot_eval: boolean, if True plot training progression 
    :custom_callbacks: boolean, if True it use Metrics() class to monitor training
    :verbose: int
    :name: string
    :plt_figsize: tuple of int or float
    :plt_top: float or int, the position of the top edge of the subplots
    :plt_save: boolean, if True it save the final plot in OUT_PATH directory as .png
    """  

    start_time = datetime.now()
    print(f"Performing {name} CV")

    # Initialize df to store predictions 
    train_yprob_vec, train_ypred_vec, train_ytrue_vec = np.array([]), np.array([]), np.array([])
    val_yprob_vec, val_ypred_vec, val_ytrue_vec = np.array([]), np.array([]), np.array([])
    histories = []
    histories_custom = []

    # CV by chromosomes
    val_chr_lst = []
    percentage_pos_lst = []
    val_chr_folds = cv_split(metadata, n_val_chr, cv_seed)

    # Split according to the fold
    for i, val_chr in enumerate(val_chr_folds):
        val_chrs = ', '.join(str(i) for i in val_chr)
        val_chr_lst.append(val_chrs)
        print(f"\n> Starting CV iteration {i+1}, valid chrs = {val_chrs}") 
        ival = metadata.chr.isin(val_chr)
        X_train, X_val = xtrain[~ival], xtrain[ival]
        y_train, y_val = ytrain[~ival], ytrain[ival]

        # Get percentage positive samples
        total_train, neg_train, pos_train, total_val, neg_val, pos_val = get_no_samples(y_train, y_val, verbose=0, return_train_only=False)
        percentage_pos_lst.append(round(pos_val/total_val, 2))

        # Define the model
        model = cnn_model(print_name=False)
        if i == 0:
            print(model.summary())
        
        # Train and predict 
        if custom_callbacks:
            print("\nTraining model using custom callbacks:")
            i_train_output, i_val_output, i_history, i_history_custom = cnn_train_pred(X_train, y_train, X_val, y_val, 
                                                                                    model=model, epochs=epochs, class_weight=class_weight, 
                                                                                    batch_size=batch_size, plot_eval=plot_eval, norm=norm,
                                                                                    upsampling=upsampling, verbose=verbose, custom_callbacks=custom_callbacks,
                                                                                    reshape_to_tensor=reshape_to_tensor, early_stopping=early_stopping,
                                                                                    n_features=n_features, name=f"{name}_cv_{i+1}")
        else:
            print("\nTraining model:")
            i_train_output, i_val_output, i_history = cnn_train_pred(X_train, y_train, X_val, y_val, 
                                                                model=model, epochs=epochs, class_weight=class_weight, 
                                                                batch_size=batch_size, plot_eval=plot_eval, norm=norm,
                                                                upsampling=upsampling, verbose=verbose, custom_callbacks=custom_callbacks,
                                                                reshape_to_tensor=reshape_to_tensor, early_stopping=early_stopping,
                                                                n_features=n_features, name=f"{name}_cv_{i+1}")

        # Store prediction on training data
        train_yprob_vec = np.concatenate((train_yprob_vec, i_train_output["yprob"]))
        train_ypred_vec = np.concatenate((train_ypred_vec, i_train_output["ypred"]))
        train_ytrue_vec = np.concatenate((train_ytrue_vec, i_train_output["ytrue"]))
        # Store prediction on validation data
        val_yprob_vec = np.concatenate((val_yprob_vec, i_val_output["yprob"]))
        val_ypred_vec = np.concatenate((val_ypred_vec, i_val_output["ypred"]))
        val_ytrue_vec = np.concatenate((val_ytrue_vec, i_val_output["ytrue"]))
        # Store history
        histories.append(i_history)
        if custom_callbacks:
            histories_custom.append(i_history_custom)

    # Stack training and validation predictions into two panda df
    train_output = pd.DataFrame({"yprob": train_yprob_vec, "ypred": train_ypred_vec,"ytrue": train_ytrue_vec})
    val_output = pd.DataFrame({"yprob": val_yprob_vec, "ypred": val_ypred_vec,"ytrue": val_ytrue_vec})

    # Evaluate
    if custom_callbacks:
        plot_loss_cnn_cv(histories_custom, val_chrs=val_chr_lst, percentage_pos=percentage_pos_lst, 
                        figsize=plt_figsize, top=plt_top, save=plt_save, name=name)

    print(f"\n>> {name} CV final report")
    print_output_eval(train_output, val_output)

    # Report time 
    print(f"Duration: {datetime.now() - start_time}")
    return train_output, val_output #, histories, histories_custom



###### Prediction of new data ######

def final_pred(xtest, ytest, metadata, scaler_model_name, 
               algo="cnn", filename=FILENAME_MODEL, 
               model_path=MODEL_PATH, n_iter=7, 
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
    :plt_save: boolean, if True it save the final plot in OUT_PATH directory as .png
    """  

    # Initialize   
    start_time = datetime.now()
    print(f"Performing prediction with {scaler_model_name} model")   
    print("Algorithm =", algo.upper())
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
        scaler_filepath = f"{model_path}{name}_scaler_{filename}.pkl" 
        print(f"Loading scaler:\t{scaler_filepath}")  
        scaler = pickle.load(open(scaler_filepath, "rb")) 

        # Normalize
        i_xtest = xtest.copy()
        if norm:
            i_xtest = pd.DataFrame(scaler.transform(i_xtest))  
        
        # Load the model
        model_filepath = f"{model_path}{name}_model_{filename}"
        if algo == "cnn":
            model_filepath = f"{model_filepath}.h5"
            model = load_model(model_filepath)  
            # Rehape
            i_xtest = i_xtest.values.reshape((i_xtest.shape[0], -1, 1))
        if algo == "lgbm":
            model_filepath = f"{model_filepath}.txt"
            model = lgb.Booster(model_file = model_filepath)
        print(f"Loading model:\t{model_filepath}")
        
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
        # Final plot
        names_index = [f"{algo.upper()} {i+1}" for i in range(n_iter)]
        names_index.append("Final model")
        output_lst.append(final_output)
        eval_heatmap(output_lst, index=names_index, title=f"{algo.upper()} Test", filename= f"{scaler_model_name}_test", 
                    save=plt_save, vmin=0.5, vmax=0.9, figsize=(4.7, 3.5))
    # Report time 
    print(f"Duration: {datetime.now() - start_time}")
    return final_output