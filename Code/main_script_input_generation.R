
## SUBSTITUTE ALL CALL.CBIND/RBIND with APPLY


######################### General functions ######################### 


# Function to split positive and negative profiles and metadata
split_pos_neg <- function(input_profiles, input_metadata){
  output <- list()
  # Positive
  pos_windows <- list()
  pos_windows$profiles <- input_profiles[input_metadata$label == 1,]  
  if ("label" %in% colnames(pos_windows$profiles)){
    pos_windows$profiles <- pos_windows$profiles %>% select(-label)
  }
  pos_windows$metadata <- input_metadata[input_metadata$label == 1,]
  output$pos <- pos_windows 
  # Negative
  neg_windows <- list()
  neg_windows$profiles <- input_profiles[input_metadata$label == 0,]  
  if ("label" %in% colnames(neg_windows$profiles)){
    neg_windows$profiles <- neg_windows$profiles %>% select(-label)
  }
  neg_windows$metadata <-input_metadata[input_metadata$label == 0,]
  output$neg <- neg_windows 
  # Size
  if (nrow(input_profiles) != nrow(input_metadata)){
    stop("Input profiles and metadata have different length")
  }
  print(paste("Size total windows profile:", nrow(input_profiles)))
  print(paste("Size positive profiles:", nrow(pos_windows$profiles)))
  print(paste("Size negative profiles:", nrow(neg_windows$profiles)))
  return(output)
}

# Report time execution
report_time_execution <- function(fun){
  start_time <- Sys.time()
  output <- fun
  print(Sys.time() - start_time)
  return(output)
}

# Convert CAGE profiles to GRanges object
profiles_to_granges <- function(windows){
  win_width = ATAC_BP_EXT*2+1
  gr = GRanges(seqnames=windows$metadata$chr, 
               IRanges(windows$metadata$atac_start,
                       width=rep(win_width, nrow(windows$metadata))),
               tss_total_score = windows$metadata$total_score,
               tss_max_score = windows$metadata$max_score)
  return(gr)
}


######################### Extraction of windows profiles #########################


###### Extraction ######


# Return score if the position is present, 0 otherwise
get_count_vector <- function(pos, df){
  if (pos %in% df$atac_relative_pos) {
    return(df$score[which(df$atac_relative_pos == pos)])
  } else {return(0)}
}

# Return score vector for each position for both strands
get_count_vector_both_strands <- function(df){
  plus_count_vector <- sapply(1:len_vec, get_count_vector, df = df[df$strand == "+",])
  minus_count_vector <- sapply(1:len_vec, get_count_vector, df = df[df$strand == "-",])
  return(c(plus_count_vector, minus_count_vector))
}

# Return the CAGE profiles of the (CAGE) overlapping ATAC regions
get_chr_windows_profiles <- function(cage_granges, atac_granges, chr){
  # Select chromosome
  cage_granges <- cage_granges[seqnames(cage_granges) == chr]
  atac_granges <- atac_granges[seqnames(atac_granges) == chr]
  # Add all information into one df
  overlaps <- findOverlaps(cage_granges, atac_granges)                   # Index of overlapping CAGE fragment
  # Check if there are ATAC positive windows overlapping CAGE data
  if (length(overlaps) > 0){                                                   
    df <- cage_granges[queryHits(overlaps)]                              # Keep only overlapping CAGE data
    df$index_overlapping_atac <- subjectHits(overlaps)                   # Add index of overlapping ATAC   
    df %>% as_tibble() %>%                                               # Add ATAC start site and relative position
      mutate(atac_start = start(atac_granges[subjectHits(overlaps)]),
             atac_relative_pos = start - atac_start + 1) -> df
    # Extract profiles of each (CAGE) overlapping ATAC region
    profiles <- by(data = df, 
                   INDICES = df$index_overlapping_atac, 
                   FUN = function(x){get_count_vector_both_strands(x)})
    profiles <- data.frame(do.call(rbind, profiles))
    colnames(profiles) <- c(paste("Plus_", 1:len_vec, sep = ""), paste("Minus_", 1:len_vec, sep = ""))
    # Add metadata information
    profiles <- profiles %>% mutate(atac_start = start(atac_granges[as.numeric(rownames(profiles))]),
                                    chr = chr) %>% relocate(c(chr, atac_start), .before = Plus_1) 
    return(profiles)
  } 
  else {
    print(paste(chr, "contains no overlapping windows"))
    return(NULL)
  }
}

# Return the windows profiles for all chromosomes
get_windows_profiles <- function(cage_granges, atac_granges){
  chromosomes <- unique(seqnames(cage_granges))
  list_chr_profiles <- lapply(chromosomes, function(x) 
  {get_chr_windows_profiles(cage_granges = cage_granges, 
                            atac_granges = atac_granges,
                            chr = x)})
  output_list <- list()
  output_list$profiles <- data.frame(do.call(rbind, list_chr_profiles))  
  output_list$metadata <- output_list$profiles %>% select(chr, atac_start)
  output_list$profiles <- output_list$profiles %>% select(-chr, -atac_start)
  return(output_list)
}

# Remove the ranges overlapping non reliable genomic regions
filter_blacklist <- function(ranges, blacklist){
  overlapping_blacklist <- queryHits(findOverlaps(ranges, blacklist))
  paste("There are", length(overlapping_blacklist), "Negative ranges overlapping the blacklist")
  if (length(overlapping_blacklist) > 0){
    ranges <- ranges[-overlapping_blacklist]}
  return(ranges)
}

# Remove empty profiles
filter_empty_profiles <- function(windows){
  non_overlapping_ranges <- apply(windows$profiles, 1, sum) == 0
  if (sum(non_overlapping_ranges) > 0){
    print(paste("Removing", sum(non_overlapping_ranges), "non-overlapping ranges"))
    windows$profiles <- windows$profiles[!non_overlapping_ranges,]
    windows$metadata <- windows$metadata[!non_overlapping_ranges,]}
  return(windows)
}


###### Remove ATAC-Seq positive regions intra overlaps ######


## Compute some measure of the rank and use it to remove the overlaps: total cage score (1), max TSS score (2)

# The following function take as input the indexes of overlapping ranges and 
# it returns the index of the ranges with largest total score
get_index_largest_score <- function(ranges_indexes, metadata){
  clash_total_score <- sapply(ranges_indexes, function(x) 
    metadata$total_score[x])
  return(ranges_indexes[which.max(clash_total_score)])
} 


remove_overlaps_by_rank <- function(windows){
  # Add measures of rank
  metadata_granges <- GRanges(seqnames = windows$metadata$chr, 
                              ranges = IRanges(start = windows$metadata$atac_start, width = len_vec))
  windows$metadata <- windows$metadata %>% mutate(total_score = apply(windows$profiles, 1, sum),
                                                  max_score = apply(windows$profiles, 1, max))
  # Get index largest score by overlaps                      
  overlaps <- as.data.frame(findOverlaps(metadata_granges))
  top_rank_index <- by(overlaps, INDICES = overlaps$queryHits,                              # The function using "by" remove the overlaps keeping the range with highest rank (large score)
                       function(x){                                                         # It takes the output of findoverlaps and for each range it return the index of the overlapping 
                         get_index_largest_score(x$subjectHits, windows$metadata)})         # range with largest score  
  top_rank_index <- unique(do.call(cbind, list(top_rank_index)))
  # Keep only ranges with highest rank 
  windows$profiles <- windows$profiles[top_rank_index,]
  windows$metadata <- windows$metadata[top_rank_index,]
  return(windows)
}


###### Filtering and sampling ###### 


# Filter the negative profiles by CAGE signal 
windows_profiles_filter <- function(windows,
                                    threshold = 1, fun = max){
  filter <- apply(windows$profiles, 1, fun) > threshold
  windows$profiles <-  windows$profile[filter,]
  windows$metadata <- windows$metadata[filter,]
  return(windows)
}

# Sample from negative set
windows_sampling <- function(windows, size){
  if (nrow(pos_windows$metadata) > nrow(windows$metadata)){
    print(paste0("WARNING: The negative set (", as.character(nrow(windows$metadata)) ,
                 ") is smaller than the positive one (", as.character(nrow(pos_windows$metadata)), ")"))
  }else{
    uniform_sampling <- runif(size, 1, nrow(windows$metadata))
    windows$metadata <- windows$metadata[uniform_sampling,]
    windows$profiles <- windows$profiles[uniform_sampling,]
    return(windows) 
  }
}




######################### Normalization and forward and reverse subtraction ######################### 


# Apply forward minus reverse subtraction 
strands_norm_subtraction <- function(vec){
  # Subtract the forward and reverse signal
  p <- as.numeric(vec[1:len_vec])
  m <- as.numeric(vec[(len_vec+1):(len_vec*2)])
  # Normalized strand subtraction
  return ((p - m) / max(abs(c(p, m))))
}

strands_norm_subtraction_all_windows <- function(windows){
  # Apply normalized forward and reverse subtraction to all windows
  p_min_m_norm_df <- as.data.frame(t(apply(windows, 1, strands_norm_subtraction)))
  pos <- seq(1, ncol(p_min_m_norm_df))
  colnames(p_min_m_norm_df) <- paste0("Pos", pos - ATAC_BP_EXT - 1)
  return(p_min_m_norm_df)
}


######################### Merge the profiles from timepoints or replicates #########################


# Merge profiles and metadata from different replicates or timepoints
merge_files <- function(path, print_details=TRUE, merging="timepoints"){
  # Extract the filename of the replicates profiles and metadata 
  replicate_lst <- list.files(path)
  complete_data_lst <- replicate_lst[!grepl("test|train", replicate_lst)]
  profiles_lst <- complete_data_lst[grep("profiles", complete_data_lst)]
  metadata_lst <- complete_data_lst[grep("metadata", complete_data_lst)]
  output <- list()
  # Merge dataframes
  if (merging=="replicates"){
    metadata <- lapply(seq(length(metadata_lst)), 
                       function(x) read.csv(paste0(path, "/", metadata_lst[x]), header = TRUE) %>% mutate(rep = x))    
  } else if (merging=="timepoints"){
    metadata <- lapply(seq(length(metadata_lst)), 
                       function(x) read.csv(paste0(path, "/", metadata_lst[x]), header = TRUE) %>% 
                         mutate(timepoint = str_split(metadata_lst[x], "_")[[1]][7]))
  } else {stop("The argument \"merging\" must include \"replicates\" or \"timepoints\"")}
  profiles <- lapply(seq(length(profiles_lst)), 
                     function(x) read.csv(paste0(path, "/", profiles_lst[x]), header = TRUE)) 
  # Plot maximum and total score
  if (print_details == TRUE){
    print("Metadata:")
    sapply(seq(length(metadata_lst)), function(x) print(paste(metadata_lst[[x]], dim(metadata[[x]])[1], dim(metadata[[x]])[2]))) 
    print("Profiles:")
    sapply(seq(length(profiles_lst)), function(x) print(paste(profiles_lst[[x]], dim(profiles[[x]])[1], dim(profiles[[x]])[2]))) 
  }
  output$metadata <- do.call("rbind", metadata)
  output$profiles <- do.call("rbind", profiles)
  return(output)
}




######################### Profiles exploration ######################### 


###### Forward and reverse concatenated ###### 


# Plot chr distribution
plot_chr_distribution <- function(window_metadata, 
                                  title = "Chr distribution",
                                  save = FALSE,
                                  path = ""){
  krows = round(nrow(window_metadata)/1000)
  window_metadata %>% group_by(chr) %>% count() %>% 
    ggplot(aes(x = factor(chr, levels = paste("chr", c(1:22, "X", "Y"), sep="")), 
               y = n)) +
    geom_bar(stat = "identity", col = "black", fill = brewer.pal(8,"Dark2")[1]) +
    labs(title = paste0("Chr distribution (", krows, "k)"),
         x = "Chromosome", y = "N. windows") +
    theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust=1, size=12)) -> plot
  if (save) {ggsave(path, plot, height = 5, dpi = 300) }
  return(plot)
}

# Get distribution of CAGE coverage for ATAC-Seq peak relative positions
get_cage_distribution_by_peak_position <- function(peaks_profile_df){
  apply(peaks_profile_df, 2, sum) %>% as_tibble() %>% 
    mutate(pos = c(-ATAC_BP_EXT:ATAC_BP_EXT, -ATAC_BP_EXT:ATAC_BP_EXT), 
           strand = c(rep("+", len_vec), rep("-", len_vec))) %>% 
    rename(score= value) %>% relocate(score, .after = strand) %>%
    mutate(score = ifelse(strand == "-", -score, score))
}

# Plot distribution of CAGE coverage for ATAC-Seq peak relative positions
plot_cage_distribution_by_peak_position <- function(peaks_profile_df, save=FALSE, 
                                                    path="", title=NA){
  # Get cage distribution by position
  cage_distribution_by_peak_position <- get_cage_distribution_by_peak_position(peaks_profile_df)
  krows = round(nrow(peaks_profile_df)/1000)
  # Plot
  if (is.na(title)){title = paste0("CAGE score distribution over ATAC-Seq peaks relative position (All chr, ", krows, "k)")}
  else {title = paste0(title, " (All chr, ", krows, "k)")}
  cage_distribution_by_peak_position %>% ggplot(aes(x = pos, y = score, color = strand)) + geom_line() +
    labs(title = title,  
         x = "Relative position to ATAC mid peaks", y = "Sum of scores over windows") + 
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + # coord_cartesian(ylim = c(250000, -250000)) + 
    theme_bw() -> plot_cage_by_peak_pos
  # Save
  if (save){ggsave(path, plot_cage_by_peak_pos, height = 5, dpi = 300)}
  return(plot_cage_by_peak_pos)
}

# Get distribution of ATAC-Seq peaks CAGE total coverage 
get_windows_total_cage_score_distribution <- function(peaks_profile_df){
  peaks_profile_df %>% 
    count(apply(peaks_profile_df, 1, sum)) %>% 
    rename(total_score = "apply(peaks_profile_df, 1, sum)", n_atac_peaks = n) %>%
    relocate(total_score, .after = n_atac_peaks) 
}

# Plot distribution of ATAC-Seq peaks CAGE total coverage (ATAC-Seq peaks number)
plot_profiles_total_score_distribution <- function(peaks_profile_df, score_filter=50, save=FALSE, path=""){
  # Get total cage score distribution
  windows_total_cage_score_distribution <- get_windows_total_cage_score_distribution(peaks_profile_df)
  # Plot
  windows_total_cage_score_distribution %>% filter(total_score < score_filter) %>% 
    ggplot(aes(x = total_score, y = n_atac_peaks)) + 
    geom_bar(stat = "identity", color = "black", fill = brewer.pal(8,"Dark2")[5]) + 
    scale_y_continuous(breaks = scales::pretty_breaks(n = 15)) +
    labs(title = "Windows profiles total score distribution", 
         y = "N. Windows profiles",
         x = "Total CAGE score",
         fill = NA) +
    theme_bw() + theme(legend.position = "none") -> windows_total_cage_score_plot
  # Save
  if (save){ggsave(path, windows_total_cage_score_plot, height = 5, dpi = 300)} 
  return(windows_total_cage_score_plot)
}

# Plot distribution of ATAC-Seq peaks CAGE total coverage (CAGE total coverage per peak)
plot_profiles_total_score_distribution_alt <- function(peaks_profile_df, save=FALSE, path=""){
  # Get total cage score distribution
  windows_total_cage_score_distribution <- get_windows_total_cage_score_distribution(pos_windows$profiles)
  # Plot
  windows_total_cage_score_distribution %>% mutate(type = "Windows profiles") %>% 
    ggplot(aes(x = type, y = total_score)) + geom_violin(fill= brewer.pal(8,"Greys")[3]) +
    geom_jitter(aes(size = n_atac_peaks, col = n_atac_peaks), alpha=0.5) + theme_bw() +
    labs(title = "Windows profiles total score distribution", 
         x = NA,
         y = "Total CAGE score",
         size = "N. Windows profiles",
         col = "") +
    facet_zoom(ylim=c(0, 250000), shrink = FALSE) +
    scale_colour_gradientn(colours = c("blue", "red"), values = c(0, 0.5, 1)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    theme(axis.title.x=element_blank()) -> windows_total_cage_score_plot
  # Save
  if (save){ggsave(path, windows_total_cage_score_plot, height = 5, dpi = 300)} 
  return(windows_total_cage_score_plot)
}

# Plot the maximum tss score of each window
plot_max_tss_score_distribution <- function(peaks_profile_df, save=FALSE, path="", 
                                            coord_xlim = c(0, 50), y_zoom=c(0, 500)){
  # Get max TS scores
  max_tss_score_distribution <- rownames_to_column(peaks_profile_df, var = "atac_start") %>% 
    mutate(max_tss_score = apply(peaks_profile_df, 1, max)) %>% 
    select(atac_start, max_tss_score) %>% group_by(max_tss_score) %>% count()
  # Plot
  max_tss_score_distribution %>% ggplot(aes(x = max_tss_score, y = n, fill = "red")) + 
    geom_bar(stat = "identity", color = "black", fill = brewer.pal(8,"Dark2")[3]) + 
    facet_zoom(ylim=y_zoom, shrink = FALSE) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 15)) +
    coord_cartesian(xlim = coord_xlim) +
    labs(title = "Max TSS score distribution", 
         y = "N. windows profiles",
         x = "Max TSS score",
         fill = NA) +
    theme_bw() +
    theme(legend.position = "none") -> max_tss_score_plot
  if (save){ggsave(path, max_tss_score_plot, height = 4.5, dpi = 300)}
  return(max_tss_score_plot)  
}

## Exploration of different chromosomes score distribution

# Get the distribution of CAGE score over one chr windows positions 
get_score_distribution_by_pos_one_chr <- function(list_windows, chr){
  windows_profile <- list_windows$profiles[list_windows$metadata == chr,]
  score_distribution <- get_cage_distribution_by_peak_position(windows_profile)
  score_distribution$chr = chr
  return(score_distribution)  
}

# Get the distribution of CAGE score for all chr
get_score_distribution_by_pos_all_chr <- function(list_windows){
  list_df <- lapply(unique(list_windows$metadata$chr), function(x){
    get_score_distribution_by_pos_one_chr(list_windows, as.character(x))}) 
  return(data.frame(do.call(rbind, list_df)))
}

# Plot score distribution by position by chr
plot_score_distribution_by_pos_by_chr <- function(list_windows, scales="free_y", coord_ylim=c(-17000, 17000), 
                                                  save=FALSE, path="", title=NA){
  # Get score distribution by chr
  windows_score_distribution_by_pos_all_chr <- get_score_distribution_by_pos_all_chr(list_windows)
  krows = round(nrow(list_windows$profiles)/1000)
  # Plot
  if (is.na(title)){title = paste("CAGE score distribution over ATAC-Seq peaks relative position (", krows, "k)", sep="")}
  else {title = paste0(title, " (All chr, ", krows, "k)")}
  windows_score_distribution_by_pos_all_chr %>% 
    ggplot(aes(x = pos, y = score, color = strand)) + geom_line() +
    labs(title = title,
         x = "Relative position to ATAC mid peaks", y = "Sum of scores over windows") +
    facet_wrap(~factor(chr,
                       levels = paste("chr", c(1:22, "X", "Y"), sep="")), scales=scales) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) + theme_bw() -> windows_score_by_pos_by_chr
  if (scales=="fixed"){
    windows_score_by_pos_by_chr <- windows_score_by_pos_by_chr <- windows_score_by_pos_by_chr + coord_cartesian(ylim = coord_ylim)}
  # Save
  if (save){ggsave(path, windows_score_by_pos_by_chr, height = 7, width = 12, dpi = 300)}
  return(windows_score_by_pos_by_chr)
}

## Exploration of different chromosomes profiles
plot_set_profiles <- function(windows, chr="all", details = FALSE, title = "",
                              coord_ylim = c(-50, 50), scales = "free_y",
                              range = 1:56, sort = FALSE, save = FALSE, path=""){
  # Get number of windows to plots
  if (max(range) > nrow(windows$profiles)){range = 1:nrow(windows$profiles)}
  # Compute max and total coverage for each window
  if (chr != "all") {windows_profile <- windows$profiles[windows$metadata$chr == chr,]} 
  else {windows_profile <- windows$profiles}
  windows_profile %>% 
    mutate(total_coverage = apply(windows_profile, 1, sum),
           max_coverage = apply(windows_profile, 1, max)) -> windows_profile
  # Add details (minimum TSS score and total score of the window)
  if (details){
    row_name <- paste("w_", 1:nrow(windows_profile), "\n(T=", 
                      as.character(windows_profile$total_coverage), ", \nM=", 
                      windows_profile$max_coverage, ")")} 
  else {row_name <- paste0("w_", 1:nrow(windows_profile))}
  windows_profile %>%
    mutate(window = row_name) %>%
    relocate(c(window, total_coverage), .before = Plus_1) -> temp
  if (sort){temp %>% arrange(desc(total_coverage)) %>% slice(range) -> temp} 
  # Prepare for plotting
  temp <- temp %>% select(-total_coverage, -max_coverage) %>% slice(range) %>% melt()
  temp$value[grepl("^M", temp$variable)] <- -temp$value[grepl("^M", temp$variable)] 
  temp$strand[grepl("^M", temp$variable)] <- "-"
  temp$strand[grepl("^P", temp$variable)] <- "+"
  temp$variable <- as.numeric(gsub("\\D", "", temp$variable)) - ATAC_BP_EXT - 1           
  # Plot
  temp %>% ggplot(aes(x = variable, y = value, color = strand)) + geom_line() + 
    facet_wrap(~window, ncol = 8, strip.position = "left", scales = scales) +
    labs(title = paste("Windows profiles (", 
                       deparse(substitute(range)), ", ", chr, ")", sep = ""), 
         y = "TSS score",
         x = "Relative position to ATAC-Seq mid peak",
         fill = NA) + 
    theme_bw() -> plot
  if (scales=="fixed") {plot <- plot + coord_cartesian(ylim = coord_ylim)}
  if (save){ggsave(path, plot, height = 9, width = 20, dpi = 300)}
  return(plot)
}


###### Forward and reverse subtracted ###### 


# Get profiles distribution all chromosomes
get_subt_cage_distribution_by_peak_position <- function(input_profiles){
  apply(input_profiles, 2, sum) %>% as_tibble() %>% 
    mutate(pos = c(-ATAC_BP_EXT:ATAC_BP_EXT)) %>% 
    rename(score = value)
}

# Plot profiles distribution all chromosomes
plot_subt_score_distribution_by_pos <- function(input_profiles, save=FALSE, path="", title=NA){
  # Get score distribution by position
  subt_score_distribution_by_pos <- get_subt_cage_distribution_by_peak_position(input_profiles)
  krows = round(nrow(input_profiles)/1000)
  # Plot
  if (is.na(title)){title = paste0("CAGE score distribution over ATAC-Seq peaks relative position (All chr, ", krows, "k)")}
  else {title = paste0(title, " (All chr, ", krows, "k)")}
  subt_score_distribution_by_pos %>% ggplot(aes(x = pos, y = score)) + geom_line(color="deepskyblue") +
    labs(title = title,  
         x = "Relative position to ATAC mid peaks", y = "CAGE signal (Forward - Reverse)") + 
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + # coord_cartesian(ylim = c(250000, -250000)) + 
    theme_bw() -> plot_subt_score_distribution_by_pos
  if (save){
    ggsave(path, plot_subt_score_distribution_by_pos, height = 5, dpi = 300) 
  }
  return(plot_subt_score_distribution_by_pos)
}

## Plot profiles distribution by chromosome

# Get the distribution of CAGE score over one chr windows positions 
get_subt_score_distribution_by_pos_one_chr <- function(list_windows, chr){
  windows_profile <- list_windows$profiles[list_windows$metadata == chr,]
  score_distribution <- get_subt_cage_distribution_by_peak_position(windows_profile)
  score_distribution$chr = chr
  return(score_distribution)  
}

# Get the distribution of CAGE score for all chr
get_subt_score_distribution_by_pos_all_chr <- function(list_windows){
  list_df <- lapply(unique(list_windows$metadata$chr), function(x){
    get_subt_score_distribution_by_pos_one_chr(list_windows, as.character(x))}) 
  return(data.frame(do.call(rbind, list_df)))
}

# Plot the distribution of CAGE score by position for each chr
plot_subt_score_distribution_by_pos_by_chr <- function(list_windows, scales="free_y", coord_ylim=c(-17000, 17000), save=FALSE, path=""){
  # Get subtracted score distribution by position for each chromosome
  windows_subt_score_distribution_by_pos_all_chr <- get_subt_score_distribution_by_pos_all_chr(list_windows)
  krows = round(nrow(list_windows$profiles)/1000)
  # Plot
  windows_subt_score_distribution_by_pos_all_chr %>% 
    ggplot(aes(x = pos, y = score)) + geom_line(color="deepskyblue") +
    labs(title = paste("Normalized set CAGE score distribution over ATAC-Seq peaks relative position (", krows, "k)", sep=""),
         x = "Relative position to ATAC mid peaks", y = "Sum of scores over windows") +
    facet_wrap(~factor(chr,
                       levels = paste("chr", c(1:22, "X", "Y"), sep="")), scales=scales) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) + theme_bw() -> windows_score_by_pos_by_chr
  if (scales=="fixed"){
    windows_score_by_pos_by_chr <- windows_score_by_pos_by_chr <- windows_score_by_pos_by_chr + coord_cartesian(ylim = coord_ylim)}
  # Save
  if (save){ggsave(path, windows_score_by_pos_by_chr, height = 7, width = 12, dpi = 300)}
  return(windows_score_by_pos_by_chr)
}

# Exploration of different chromosomes profiles
plot_subt_set_profiles <- function(windows, chr="all", details = FALSE, title = "",
                                   coord_ylim = c(-50, 50), scales = "free_y",
                                   range = 1:56, sort = FALSE, save = FALSE, path=""){
  # Get number of windows to plots
  if (max(range) > nrow(windows$profiles)){range = 1:nrow(windows$profiles)}
  # Compute max and total coverage for each window
  if (chr != "all") {windows_profile <- windows$profiles[windows$metadata$chr == chr,]} 
  else {windows_profile <- windows$profiles}
  windows_profile %>% 
    mutate(total_coverage = apply(windows_profile, 1, sum),
           max_coverage = apply(windows_profile, 1, max)) -> windows_profile
  # Add details (minimum TSS score and total score of the window)
  if (details){
    row_name <- paste("w_", 1:nrow(windows_profile), "\n(T=", 
                      as.character(windows_profile$total_coverage), ", \nM=", 
                      windows_profile$max_coverage, ")")} 
  else {row_name <- paste0("w_", 1:nrow(windows_profile))}
  windows_profile %>%
    mutate(window = row_name) -> temp
  if (sort){temp %>% arrange(desc(total_coverage)) %>% slice(range) -> temp} 
  # Prepare for plotting
  temp <- temp %>% select(-total_coverage, -max_coverage) %>% slice(range) %>% melt()
  temp$variable <- as.numeric(gsub("\\D", "", temp$variable))
  # Plot
  temp %>% ggplot(aes(x = variable, y = value)) + geom_line(color="deepskyblue") + 
    facet_wrap(~window, ncol = 8, strip.position = "left", scales = scales) +
    labs(title = paste("Windows profiles (", 
                       deparse(substitute(range)), ", ", chr, ")", sep = ""), 
         y = "TSS score",
         x = "Relative position to ATAC-Seq mid peak",
         fill = NA) + 
    theme_bw() -> plot
  if (scales=="fixed") {plot <- plot + coord_cartesian(ylim = coord_ylim)}
  if (save){ggsave(path, plot, height = 9, width = 20, dpi = 300)}
  return(plot)
}
