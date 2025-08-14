#!/usr/bin/env bash

# upon error, stop script
set -e

# load the activate_environment
source config.sh

function main() {

  activate_environment

  # 1st CLI argument is the input directory
  local input_directory=$(realpath "$1")
  # 2nd CLI argument is the output directory
  local output_directory=$(realpath "$2")
  # 3rd is the path to a log file
  local log_file="${3:-/dev/null}"

  # Make the output directory, (Comment out if the directory will already be there.)
  mkdir "$output_directory"

  pushd . > /dev/null
  # This will change back to the starting directory when it receives certain signals
  trap 'popd > /dev/null' EXIT SIGKILL SIGINT

  # the software must be run from this directory, therefore use absolute paths
  cd chemoTimelinesBaselineSystem/timelines

  # TQDM_DISABLE=1
  java -cp instance-generator/target/instance-generator-5.0.0-SNAPSHOT-jar-with-dependencies.jar \
    org.apache.ctakes.core.pipeline.PiperFileRunner \
    -p org/apache/ctakes/timelines/pipeline/Timelines \
    -v "$conda_environment"/bin/python \
    -i "$input_directory" \
    -o "$output_directory" \
    -l org/apache/ctakes/dictionary/lookup/fast/bsv/Unified_Gold_Dev.xml \
    --pipPbj yes  2>&1 | tee -a "$log_file"

  
}

main "$@"