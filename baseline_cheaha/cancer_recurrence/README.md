

This running the Cancer Recurrence on the baseline software.

This script takes the JSON file and splits it up into batches of <700 .txt files.
cancer_recur_json_to_txt.py

The next step is to run the baseline software batches. Using this script:  run_cancer_recur_batches.sh

This uses signals and searches the log files detect when the process has completed. It is programmed to 
go through directories with "batch0" through "batch67"
