#!/usr/bin/env Rscript

##### GENOME WIDE PROFILES EXTRACTION #####

writeLines("\nRUNNING PROFILES EXTRACTION")

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
  
  #library(ggforce)            
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
parser$add_argument("-s", "--step", default = 5, type="integer", help = "step to perform genome wide exraction")
parser$add_argument("-F", "--format", default = "gz",  help = "format of the CAGE input file")
args <- parser$parse_args()



### Initialize

writeLines("\nExtraction initialization..")
# print(paste("Input CAGE:", INPUT))
# print(paste("Filename output:", FILENAME))
# print(paste("Output directory:", OUTPUT))
# print(paste("Step:", STEP))
# print(paste("Format:", FORMAT))

# General
STEP <- args$step
FORMAT <- args$format

# Input and output
INPUT <- args$input_dir
FILENAME <- args$filename
OUTPUT <- args$output_dir

# Fixed
ATAC_BP_EXT <- 150
len_vec <- ATAC_BP_EXT * 2 + 1


### Load data

# Load CAGE
CAGE_granges <- import(paste0(INPUT), format=FORMAT)
genome(CAGE_granges) <- "hg19"

# Remove non standard chromosomes
chromosomes <- paste0("chr", c(seq(1:22), "X", "Y"))
CAGE_granges <- CAGE_granges[seqnames(CAGE_granges) %in% chromosomes]
seqlevels(CAGE_granges) = chromosomes
genome(CAGE_granges) <- "hg19"


### Get running genomic ranges with steps of 5 (default) bp

# Get running genomic ranges for all chrs
writeLines("\nSelecting regions across the genome..")
genome_wide_granges <- report_time_execution(get_genome_wide_ranges(CAGE_granges, step=STEP))
paste("Total selected regions:", length(genome_wide_granges))



### Extract the profiles

# Genome wide extraction of the CAGE profiles
writeLines("\nExtracting profiles:")
windows <- report_time_execution(get_windows_profiles(CAGE_granges, genome_wide_granges))
paste("Extracted profiles:", nrow(windows$metadata))



### Filter by CAGE signal

# Filter by minimal CAGE requirement (TSS with atleast 2)
windows <- windows_profiles_filter(windows)
paste("Extracted profiles after CAGE filtering:", nrow(windows$metadata))



### Process the profiles
writeLines("\nProcessing extracted profiles..")
windows_subt <- list()
windows_subt$profiles <- report_time_execution(strands_norm_subtraction_all_windows(windows$profiles))
windows_subt$metadata <- windows$metadata



### Export extracted profiles

# Export
writeLines("Exporting extracted profiles..")
write_csv(windows$profiles, paste0(OUTPUT, FILENAME, "/profiles_", FILENAME, ".csv"))
write_csv(windows$metadata, paste0(OUTPUT, FILENAME, "/metadata_", FILENAME, ".csv"))



### Export extracted processed profiles

# Export
writeLines("Exporting processed extracted profiles..")
write_csv(windows_subt$profiles, paste0(OUTPUT, FILENAME, "/profiles_", FILENAME, "_subtnorm.csv"))
write_csv(windows_subt$metadata, paste0(OUTPUT, FILENAME, "/metadata_", FILENAME, "_subtnorm.csv"))



### Explore extracted profiles
writeLines("\nPlotting extracted profiles")

# Plot chr distribution
plot_chr_distribution(windows$metadata,
                      save=TRUE,
                      path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/chr_distribution.png"))

# Plot the distribution of CAGE coverage for ATAC-Seq peak relative positions
plot_cage_distribution_by_peak_position(windows$profiles, save=TRUE,
                                        path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/cage_by_peak_pos.png"))

# Plot distribution of ATAC-Seq peaks CAGE total coverage (ATAC-Seq peaks number)
plot_profiles_total_score_distribution(windows$profiles, save=TRUE,
                                       path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/windows_total_cage_score.png"))


# Plot the maximum tss score of each window
plot_max_tss_score_distribution(windows$profiles, save=TRUE, y_zoom=c(0, 5000),
                                path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/max_tss_score_plot.png"))

# Exploration of different chromosomes score distribution
plot_score_distribution_by_pos_by_chr(windows, save=TRUE,
                                      path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/windows_score_by_pos_by_chr.png"))

# Plot some profiles
#plot_set_profiles(windows, chr="chr1", sort=FALSE, save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles1.png"))    <- CHECK AND FIX
# plot_set_profiles(windows, chr="chr11", sort=FALSE, save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles11.png"))
# plot_set_profiles(windows, chr="chr2", sort=FALSE, save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles2.png"))
# plot_set_profiles(windows, chr="chr3", sort=FALSE, save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles3.png"))
# plot_set_profiles(windows, chr="chr4", sort=FALSE, save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles4.png"))



### Explore processed profiles

writeLines("Plotting processed extracted profiles")

# Plot profiles distribution all chromosome
plot_subt_score_distribution_by_pos(windows_subt$profiles, save=TRUE,
                                    path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/windows_subt_score_by_pos.png"))

# Plot profiles distribution by chromosome
plot_subt_score_distribution_by_pos_by_chr(windows_subt, save=TRUE,
                                           path=paste0(OUTPUT, FILENAME, "/plots/all_profiles/windows_subt_score_by_pos_by_chr.png"))

# Plot some profiles
#plot_subt_set_profiles(windows, chr="chr1", save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles1_subt.png"))     <- CHECK AND FIX
# plot_subt_set_profiles(windows, chr="chr11", save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles11_subt.png"))
# plot_subt_set_profiles(windows, chr="chr2", save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles2_subt.png"))
# plot_subt_set_profiles(windows, chr="chr3", save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles3_subt.png"))
# plot_subt_set_profiles(windows, chr="chr4", save=TRUE, path = paste0(OUTPUT, "plots/all_profiles/profiles4_subt.png"))









