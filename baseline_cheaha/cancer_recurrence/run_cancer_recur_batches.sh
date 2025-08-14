#!/usr/bin/env bash

# This can be ran on SLURM or interactively

#SBATCH --job-name=canrec_bl         ### Name of the job
#SBATCH --nodes=1                   ### Number of Nodes
#SBATCH --ntasks=1                  ### Number of Tasks
#SBATCH --cpus-per-task=3           ### Number of Tasks per CPU
#SBATCH --gres=gpu:1                ### Number of GPUs
#SBATCH --mem-per-cpu=16G           ### Memory required
#SBATCH --partition=pascalnodes     ### Cheaha Partition  -  pascalnodes, amperenodes
#SBATCH --time=04:00:00             ### Estimated Time of Completion, 6 hours needed for Clinical ModernBERT
#SBATCH --output=./joblog/%x_%j.out          ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=./joblog/%x_%j.err           ### Slurm Error file, %x is job name, %j is job id


# This is intended to be ran on a synconously.

# upon error, stop script
set -e

# load the activate_environment
source ../config.sh

# Change this to your system value
BROKER_DIR="$HOME/prj/chemotimelines/baseline_cheaha/mybroker"

# monitors is the log files contains specific text
function signal_baseline_when_found_done() {
  local log_file="$1"

  echo "monitoring process with signal_baseline_when_found_done()"

  while true; do
    if tail "$log_file" | grep --quiet "INFO timelines_python_pipeline - Finished writing"; then
      pid=$(pgrep -f "java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar")
      echo "This batch completed 'detected Finshed writing', send signal SIGKILL(9) to end baseline process: $(ps -p $pid -o comm=)"
      kill -s SIGKILL $pid
      break # exit out of infinite loop
    fi
    # wake up every second
    sleep 1
  done

}

function run_batch() {
  # 1st CLI argument is the input directory
  local input_directory="$1"
  # 2nd CLI argument is the output directory
  local output_directory="$2"
  # 3rd is the path to a log file
  local log_file="$3"

# activate_environment

  # Make the output directory, (Comment out if the directory will already be there.)
  mkdir --parents "$output_directory"

  pushd . > /dev/null
  # This will change back to the starting directory when it receives certain signals
#  trap 'popd > /dev/null' EXIT SIGKILL SIGINT

  # the software must be run from this directory, therefore use absolute paths
  cd ../chemoTimelinesBaselineSystem/timelines

  # TQDM_DISABLE=1
  java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
    org.apache.ctakes.core.pipeline.PiperFileRunner \
    -p org/apache/ctakes/timelines/pipeline/Timelines \
    -v "$conda_environment"/bin/python \
    -i "$input_directory" \
    -o "$output_directory" \
    -l org/apache/ctakes/dictionary/lookup/fast/bsv/Unified_Gold_Dev.xml \
    --pipPbj yes  2>&1 | tee -a "$log_file" &

  popd > /dev/null

  # give it a few seconds to start a log file
  sleep 10

  signal_baseline_when_found_done "$log_file"
}


function main() {
  # CHANGE THESE ACCORDINGLY
  input_dir="$HOME/prj/chemotimelines/baseline_cheaha/cancer_recur_batches"
  output_dir="$HOME/prj/chemotimelines/baseline_cheaha/cancer_recur_tsv"

  # activate the python environment
  activate_environment

  # start the broker in the background
  $BROKER_DIR/bin/artemis run &

  # wait for the broker to get started
  sleep 10

  # if signal occurs stop or the process ends, stop the broker
  trap '$BROKER_DIR/bin/artemis stop' EXIT SIGKILL SIGINT

  
  for i in {0..67}; do  # this was the original one, which does work if the batches do not cause errors
#  for i in {6..67}; do
#  for i in {27..67}; do
#  for i in {49..67}; do
#  for i in {50..67}; do
#  for i in {0..1}; do
#  arr=(5 48 49)
#  arr=(5 49)
#  arr=(99)
#  for i in "${arr[@]}"; do
    echo "===== Running batch${i} ====="

    run_batch "$input_dir/batch${i}" \
	    "$output_dir/batch${i}" \
	    "$output_dir/batch${i}/output.log"
  done
}

main "$@"
