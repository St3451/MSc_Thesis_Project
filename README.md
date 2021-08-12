# MSc thesis project at Andersson Lab
This repository contains the report, presentation slides, and code of my MSc thesis project that I performed at the Andersson Lab (Bioinformatics Centre, University of Copenhagen). 


# Tool brief documentation :fire:
CAGE_tool attempts to infer potential active open chromatin regions from Cap Analysis of Gene Expression (CAGE) data. 

## Setup
The setup requires Anaconda to be installed, and to overcome a conflict between python and R packages, the tool uses two dedicated Anaconda environments, to which must be assigned a specific name.

### Download
Clone the repository and decompress the models:

```
$ for file in models/*.rar; do unrar e $file; done
```

### Instal Python environment

```
$ conda create -n cage_tool_py
$ conda activate cage_tool_py
$ conda install -c anaconda tensorflow-gpu
$ conda install -c anaconda pandas
$ conda install -c conda-forge lightgbm
$ conda install -c conda-forge keras
$ conda install scikit-learn
```

### Instal R environment

```
$ conda create -n cage_tool_r
$ conda activate cage_tool_r
$ conda install -c conda-forge r-base
```

Lunch R and install the following packages:

``` r
install.packages("argparse")
install.packages("tibble")
install.packages("dplyr")
install.packages("readr")
install.packages("forcats")
install.packages("ggplot2")
install.packages("tidyr")
install.packages("gridExtra")
install.packages("reshape2")
install.packages("RColorBrewer")
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")
BiocManager::install("GenomicRanges")
BiocManager::install("rtracklayer")
```

## Description
This tool allows the user to perform a genome-wide prediction of potential active Open Chromatin Regions (OCRs) or Transcriptional Regulatory Elements (TREs) from CAGE data. To achieve the forecast, it uses gradient boosting trees (default) or convolutional neural networks, which were trained on ~5 million CAGE profiles labeled using ATAC-Seq data. First, the tool scans the genome extracting the CAGE profiles of regions having a minimum transcriptional activity (2 TSS in at least one position). Then, it predicts the probability of the extracted profiles to be potential active TREs, and it generates different plots showing the results of the analysis (e.g., average shape of the predicted profiles, their distribution across the chromosomes, and the distribution of their total CAGE score). 

## Input and output

Input:
* `<CTSS_filename>.bed.gz` (or other formats supported by `rtracklayer::import()`)

Output: 
* `profiles_<filename>.csv`
* `profiles_<filename>_subtnorm.csv`
* `metadata_<filename>.csv`
* `tres_prediction_<filename>.csv`
* `tres_prediction_UCSC_track_<filename>.bed`
* Plots

The tool takes as input a CAGE Transcription Start Site (CTSS) file and it generates different plots and five tabular files: two files including the CAGE profiles of the extracted regions (`profiles_<filename>.csv` and `profiles_<filename>_subtnorm.csv`), one file including metadata information (`metadata_<filename>.csv`) showing their genomic coordinates (chromosome and start site), one file including the predictions (`tres_prediction_<filename>.csv`), and one file including the predicted scores ready to be visualized as UCSC Genome Browser track (`tres_prediction_UCSC_track_<filename>.bed`). It outputs two files containing the profiles because one file (`profiles_<filename>.csv`) includes profiles represented as vectors of concatenated CAGE scores from forward and reverse strands. This representation is helpful for intuitive visualization of the CAGE signal on the different strands. In contrast, the other file (`profiles_<filename>_subtnorm.csv`) includes the processed profiles used as input for the machine learning algorithm to perform the prediction. These profiles are obtained by performing a normalized subtraction of the CAGE score between forward and reverse strands, which allows capturing the shift in the intensity of the CAGE signal between the strands. 

## Usage
The user simply needs to specify the path of the CAGE input file using the flag `-i, --input_cage`, and the automated pipeline will generate the directories to store the results, scan the genome extracting profiles of 351 bp, perform and store the predictions and generate the plots. 

```
$ ./tool_master_script.sh -i <my_CAGE_input_file.bed.gz>
```

It is possible to add the flag `-h, --help` to get additional information about the correct usage of the tool and all available options:

```
$ ./tool_master_script.sh -h
CAGE_tool - Attempt to infer potential active open chromatin regions from CAGE data

Usage: CAGE_tool [-i] <file> [OPTIONS]...

Required:
  -i, --input_cage          specify the path for the input CAGE data

Options:
  -h, --help                display this help and exit
  -f, --filename_out        specify the name that will be assigned to the output files        default is 'my_tres_prediction'
  -o, --output_dir          specify the directory to store the output                         default is './'
  -s, --step                specify the frequency (bp) to scan the genome                     default is '5'
  -F, --format              specify the format of the CAGE input file                         default is 'gz'
  -a, --algorithm           specify the machine learning algorithm ('cnn' or 'lgbm')          default is 'lgbm'
  -M, --model_dir           specify the directories to load the model and the scaler          default is './models/'
  -m, --model_name          specify the filename of the model and the scaler to load          default is 'fit_on_timepoint_0_to_2_processed'
```

It is recommended to specify an output filename using the flag `-f, --filename_out`, and an output directory using the flag `-o, --output_dir`:

```
$ ./tool_master_script.sh -i <my_CAGE_input_file.bed.gz> -f <my_filename_as_output> -o <my_output_directory/>
```

By default, the tool scans genomic regions every 5 base pair, but the user can change this by adding an integer after the flag `-s, --step`:

```
$ ./tool_master_script.sh -i <my_CAGE_input_file.bed.gz> -f <my_filename_as_output> -o <my_output_directory/> -s <100>
```

By default, the format of the input CAGE file is 'gz', but that can be changed (e.g. to `bed`) by adding the flag `-F, --format`:

```
$ ./tool_master_script.sh -i <my_CAGE_input_file.bed> -f <my_filename_as_output> -o <my_output_directory/> -F <bed>
```

The user can also select the algorithm (LGBM or CNN) used for the prediction by adding `lgbm` or `cnn` after the flag `-a, --algorithm`, and can also use different trained models by adding the flag `-M, --model_dir`, which specify the model directory, and `-m, --model_name`, which set the model name.

```
$ ./tool_master_script.sh -i <my_CAGE_input_file.bed.gz> -f <my_filename_as_output> -o <my_output_directory/> -a <cnn> -M <my_model_directory/> -m <my_model_name>
```


