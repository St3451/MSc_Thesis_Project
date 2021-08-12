#!/usr/bin/env Rscript

##### VISUALIATION OF THE EXTRACTED PROFILES #####

writeLines("\nRUNNING PROFILES VISUALIZATION")

writeLines("\nImporting R libraries..")
suppressPackageStartupMessages({
  library(argparse)           
  library(rtracklayer)        
  
  library(tibble)            
  library(dplyr)              
  library(readr)              
  library(forcats)           
  library(ggplot2)            
  library(tidyr)              
  
  library(gridExtra)          
  library(GenomicRanges)      
  library(reshape2)           
  library(RColorBrewer)       
  source("tool_r_main_script.R")
})



### ARGPARSE
parser <- ArgumentParser()
parser$add_argument("-i", "--input_dir", default = "./", help = "CAGE input path")
parser$add_argument("-f", "--filename_out", default = "filename", help = "output filename")
parser$add_argument("-o", "--output_dir", default = "./", help = "output directory")
args <- parser$parse_args()



### Initialize 

writeLines("\nVisualization initialization:")
# print(paste("Input:", INPUT))
# print(paste("Filename:", FILENAME))
# print(paste("Output:", OUTPUT))

# General
FORMAT <- args$format

# Input and output
INPUT <- args$input_dir
FILENAME <- args$filename
OUTPUT <- args$output_dir

# Fixed
ATAC_BP_EXT <- 150
len_vec <- ATAC_BP_EXT * 2 + 1


### Load data

# Load profiles and metadata
windows_metadata <- read.csv(paste0(INPUT, FILENAME, "/metadata_", FILENAME, ".csv"), header = TRUE)
windows_profiles <- read.csv(paste0(INPUT, FILENAME, "/profiles_", FILENAME, ".csv"), header = TRUE)
windows_metadata_subt <- read.csv(paste0(INPUT, FILENAME, "/metadata_", FILENAME, "_subtnorm.csv"), header = TRUE)
windows_profiles_subt <- read.csv(paste0(INPUT, FILENAME, "/profiles_", FILENAME, "_subtnorm.csv"), header = TRUE)

# Load predictions 
ml_output <- read.csv(paste0(INPUT, FILENAME, "/tres_prediction_", FILENAME, ".csv"), header=TRUE) 


### Export UCSC track

# From predictions to bed file
ml_output_prob_bed <- predictions_to_bed(windows_metadata, ml_output)

# Export
ml_output_prob_bed <- ml_output_prob_bed %>% select(seqnames, start, end, score)
write_tsv(ml_output_prob_bed, paste0(OUTPUT, FILENAME, "/tres_prediction_UCSC_track_", FILENAME, ".bed"), col_names=FALSE)


### Split profiles according to predictions
pos_pred <- split_from_pred(windows_profiles, windows_metadata, ml_output, class=1)
neg_pred <- split_from_pred(windows_profiles, windows_metadata, ml_output, class=0)
pos_pred_subt <- split_from_pred(windows_profiles_subt, windows_metadata_subt, ml_output, class=1, verbose=FALSE)
neg_pred_subt <- split_from_pred(windows_profiles_subt, windows_metadata_subt, ml_output, class=0, verbose=FALSE)



### Plot concatenated profiles

writeLines("\nPlotting extracted profiles")

# Chr distribution
plot <- plot_chr_distribution_by_prediction(pos_pred$metadata, neg_pred$metadata, save=TRUE, 
                                    path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/chr_distribution.png"))

# Plot average profiles
plot_cage_distribution_by_peak_position(pos_pred$profiles, save=TRUE, title = "Pos pred CAGE score distribution over profiles positions",          
                                        path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/pos_cage_by_peak_pos.png"))
plot_cage_distribution_by_peak_position(neg_pred$profiles, save=TRUE, title = "Neg pred CAGE score distribution over profiles positions",
                                        path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/neg_cage_by_peak_pos.png"))

# Get distribution CAGE total score 
plot_profiles_total_score_distribution(pos_pred$profiles, save=TRUE, title = "Pos pred windows profiles total score distribution",          
                                       path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/pos_windows_total_cage_score.png"))
plot_profiles_total_score_distribution(neg_pred$profiles, save=TRUE, title = "Neg pred windows profiles total score distribution",   
                                       path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/neg_windows_total_cage_score.png"))

# Plot the maximum tss score of each window
plot_max_tss_score_distribution(pos_pred$profiles, save=TRUE, title="Pos pred Max TSS score distribution",                                        
                                path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/pos_max_tss_score_plot.png"))
plot_max_tss_score_distribution(neg_pred$profiles, save=TRUE, title="Neg pred Max TSS score distribution",        
                                path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/neg_max_tss_score_plot.png"))

# Plot average profiles by chr
plot_score_distribution_by_pos_by_chr(pos_pred, save=TRUE, title="Pos pred CAGE score distribution over profiles positions",                       
                                      path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/pos_windows_score_by_pos_by_chr.png"))
plot_score_distribution_by_pos_by_chr(neg_pred, save=TRUE, title="Neg pred CAGE score distribution over profiles positions",
                                      path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/neg_windows_score_by_pos_by_chr.png"))

# Plot some profiles
# plot_set_profiles(pos_pred_lgb, chr="chr1", save=TRUE, path = paste0(OUTPUT, "predicted_profiles/pos_profiles1.png")) 
# plot_set_profiles(neg_pred_lgb, chr="chr1", save=TRUE, path = paste0(OUTPUT, "predicted_profiles/neg_profiles1.png")) 



### Plot normalized subtracted profiles

writeLines("\nPlotting processed extracted profiles")

# Plot average profiles
plot_subt_score_distribution_by_pos(pos_pred_subt$profiles, save=TRUE, title="Pos pred processed CAGE signal distribution over profiles positions",        
                                    path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/pos_windows_subt_score_by_pos.png"))
plot_subt_score_distribution_by_pos(neg_pred_subt$profiles, save=TRUE,  title="Neg pred processed CAGE signal distribution over profiles positions",
                                    path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/neg_windows_subt_score_by_pos.png"))

# Plot average profiles by chr
plot_subt_score_distribution_by_pos_by_chr(pos_pred_subt, save=TRUE, title="Pos pred processed CAGE signal distribution over profiles positions",          
                                           path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/pos_windows_subt_score_by_pos_by_chr.png"))
plot_subt_score_distribution_by_pos_by_chr(neg_pred_subt, save=TRUE, title="Neg pred processed CAGE signal distribution over profiles positions",
                                           path=paste0(OUTPUT, FILENAME, "/plots/predicted_profiles/neg_windows_subt_score_by_pos_by_chr.png"))

# # Plot some profiles
# plot_subt_set_profiles(pos_pred_cnn_subtnorm, chr="chr1", save=TRUE, path = paste0(OUTPUT, "plots/predicted_profiles/pos_profiles1.png")) 
# plot_subt_set_profiles(neg_pred_cnn_subtnorm, chr="chr1", save=TRUE, path = paste0(OUTPUT, "plots/predicted_profiles/neg_profiles1.png"))


