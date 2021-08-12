#!/bin/bash

while test $# -gt 0; do
           case "$1" in
    		-h|--help)
     		    echo "CAGE_tool - Attempt to infer potential active open chromatin regions from CAGE data"
      		    echo " "
     		    echo "Usage: CAGE_tool [-i] <file> [OPTIONS]..."
     		    echo " " 
                    echo "Required:"
     		    echo "  -i, --input_cage          specify the path for the input CAGE data"
      		    echo " "
                    echo "Options:"
                    echo "  -h, --help                display this help and exit"
                    echo "  -f, --filename_out        specify the name that will be assigned to the output files        default is 'my_tres_prediction'"
		    echo "  -o, --output_dir          specify the directory to store the output                         default is './'"
      		    echo "  -s, --step                specify the frequency (bp) to scan the genome                     default is '5'" 
	            echo "  -F, --format              specify the format of the CAGE input file                         default is 'gz'"
                    echo "  -a, --algorithm           specify the machine learning algorithm ('cnn' or 'lgbm')          default is 'lgbm'" 
                    echo "  -M, --model_dir           specify the directories to load the model and the scaler          default is './models/'"
                    echo "  -m, --model_name          specify the filename of the model and the scaler to load          default is 'fit_on_timepoint_0_to_2_processed'"
		    exit 0
     		    ;;
                -i|--input_cage)
                    shift
                    input_cage=$1
                    shift
                    ;;
                -f|--filename_out)
                    shift
                    filename_out=$1
                    shift
                    ;;
                -o|--output_dir)
                    shift
                    output_dir=$1
                    shift
                    ;;
                -s|--step)
                    shift
                    step=$1
                    shift
                    ;;
                -F|--format)
                    shift
                    format=$1
                    shift
                    ;;
                -a|--algorithm)
                    shift
                    algorithm=$1
                    shift
                    ;;
		-M|--model_dir)
                    shift
		    model_dir=$1
                    shift
                    ;;
                -m|--model_name)
                    shift
                    model_name=$1
                    shift
                    ;;

                 *)
                    echo "Fatal error: $1 is not a recognized flag!"
                    exit 1
                    ;;
          esac
  done  

# Raise error if the input file is not specified
if [ -z "${input_cage}" ]; then
  echo >&2 "Fatal error: --input_cage is not set"
  exit 2
fi

# Print parameters and assign default values if not set
echo ""
echo "Input CAGE: $input_cage"; 
echo "Filename output: "${filename_out:=my_tres_prediction}"";
echo "Output directory: "${output_dir:=./}"";
echo "Step: "${step:=5}"";
echo "Algorithm: "${algorithm:=lgbm}""
echo "Model and scaler directory: "${model_dir:=./models/}""
echo "Model and scaler filename: "${model_name:=fit_on_timepoint_0_to_2_processed}""
echo "Input CAGE format: "${format:=gz}"";

# Initialize conda in the subshell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)";

## Run the scripts using dedicated environments
# Activate R env
conda activate cage_tool_r &&
# Generate directories to store result
python3 tool_create_dir.py -o $output_dir -f $filename_out &&
# Perform genome wide profiles extraction
Rscript tool_extraction.R -i $input_cage -f $filename_out -o $output_dir -s $step -F $format &&
# Activate python env
conda deactivate && conda activate cage_tool_py &&
# Perform prediction of the extracted profiles
python3 tool_prediction.py -i $output_dir -f $filename_out -o $output_dir -a $algorithm -M $model_dir -m $model_name && 
# Activate R env
conda deactivate && conda activate cage_tool_r &&
# Perform visualization of the predicted profiles
Rscript tool_visualization.R -i $output_dir -f $filename_out -o $output_dir &&
# Delete redundant metadata
rm ${output_dir}/${filename_out}/metadata_${filename_out}_subtnorm.csv
