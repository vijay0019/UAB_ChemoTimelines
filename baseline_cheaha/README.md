These are instruction to setup [End-to-end to baseline system](https://github.com/HealthNLPorg/chemoTimelinesBaselineSystem/commit/103923eddd489cacd06a305e7b9629e4a1300c88) 
provided with the for the ChemoTimelines 2025 on Cheaha.


## Notes
I took the approach to have you run each of the steps to see the output.  Make sure that nothing fails spectacularly.

The original ran on Docker container.  There were different attempts to convert this to run on Singularity version 3.5.2.
During running the the baseline system, cTAKES would take the install the Python to Java Bridge during runtime.

Singularity did not like having a container trying to install more software while running the software. 
This could not be disabled by setting the `--pipPbj no` flag.
In Singularity's container format terms `%post` is where you can install software, and  `%runscript` is where the software runs.  
`%runscript` should not install software.


## Installation
I took the Docker container and installed those on Cheaha.

Get a CPU node that is not login004 to run these instructions.

```bash
# Get a node for 2 hours, for installation
srun --ntasks=1 --time=02:00:00 --mem-per-cpu=16G --partition=interactive --job-name=instl_cbtl --pty /bin/bash
```

```bash
# 1. load Anaconda, Java 8 SDK (only needed), and CUDA 11.8.0.
module load Anaconda3 Java/1.8.0_202 CUDA/11.8.0

# 2. create the conda environment in a local directory ./cuda_118_env
# Change the name in config.sh from "./cuda_118_env" to the conda environment name you want to use.
conda create --prefix ./cuda_118_env python=3.9

# 3. activate the environment
conda activate ./cuda_118_env 

pip install --upgrade pip

# install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install other python requirements
pip install stomp.py dkpro-cassis transformers[torch] pandas tomli setuptools
```

### Downloads

This downloads and decompresses the archives to the local file system

```bash
curl -fsSL https://archive.apache.org/dist/activemq/activemq-artemis/2.19.1/apache-artemis-2.19.1-bin.tar.gz | tar xzf - -C .
curl -fsSL https://archive.apache.org/dist/maven/maven-3/3.8.9/binaries/apache-maven-3.8.9-bin.tar.gz | tar xzf - -C .
```

If you are having problems downloading due to certification verification, add the `--insecure` flag to `curl` like below:

```bash
curl -fsSL --insecure https://archive.apache.org/dist/activemq/activemq-artemis/2.19.1/apache-artemis-2.19.1-bin.tar.gz | tar xzf - -C .
curl -fsSL --insecure https://archive.apache.org/dist/maven/maven-3/3.8.9/binaries/apache-maven-3.8.9-bin.tar.gz | tar xzf - -C .
```

### Create the broker
The broker is Artemis 2.19.1 because that is the last version that uses Java 8.  (The current versions require Java 17+.)

```bash
# create the broker
# Or chose a different name than "./mybroker".
./apache-artemis-2.19.1/bin/artemis create ./mybroker --user deepphe --password deepphe --allow-anonymous

# Disable the well intended limits on filling up a system.  
sed -i 's|<max-disk-usage>90</max-disk-usage>|<max-disk-usage>100</max-disk-usage>|g' ./mybroker/etc/broker.xml
```

sed is being changing this value `<max-disk-usage>90</max-disk-usage>` from `90` to `100`  in the `mybroker/etc/broker.xml` file.

Note: If you serially want to do different runs, you only need one broker.  If you need to run to be done simultaneously,
you will need as many brokers as number of simultaneous runs.  Perhaps `mybroker1`, `mybroker2`, and so on.

### Get submodule chemoTimelinesBaselineSystem

Download the chemoTimelinesBaselineSystem repo.

```bash
# load a more recent git version
module load git

cd chemoTimelinesBaselineSystem

git submodule update --init --recursive
```

### Compile

Compile cTAKES and its dependencies.

```bash
cd timelines

../../apache-maven-3.8.9/bin/mvn -U clean package

cd ../..
```

This does produce an `[ERROR] -release is only supported on Java 9 and higher`.  That seems like it can be safely ignored.


If it ran successfully it should look something like this:
```
[INFO] META-INF/maven/org.apache.ctakes/ already added, skipping
[INFO] META-INF/DEPENDENCIES already added, skipping
[INFO] META-INF/LICENSE already added, skipping
[INFO] META-INF/NOTICE already added, skipping
[INFO] ------------------------------------------------------------------------
[INFO] Reactor Summary for txtimelines-lookup 5.0.0-SNAPSHOT:
[INFO]
[INFO] txtimelines-lookup ................................. SUCCESS [  9.005 s]
[INFO] timenorm ........................................... SUCCESS [ 33.394 s]
[INFO] instance-generator ................................. SUCCESS [01:47 min]
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  02:30 min
[INFO] Finished at: 2025-08-04T13:18:51-05:00
[INFO] ------------------------------------------------------------------------
```

If you get certificate error disable certificate verification by including the flags `-Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true`:

```bash
../../apache-maven-3.8.9/bin/mvn -Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true -U clean package

cd ../..
```


### Configuration
Edit this configuration file `config.sh`, point it to the conda environment if you did not use the default.


## Running

`exit` out of the CPU node used for installation.

Get a GPU node (exit out of the current one node used for installation):
```bash
./get_node.sh
```

This uses 3 CPUS, to try to provide a CPU for the broker, baseline software, and cTAKEs.  
There might be a case for using more CPUs.  I was only trying to make sure that this ran within 15
minutes for under 700 records.

You will interactively run this:

```bash
# start the broker as a background process, and wait maybe 30 seconds
mybroker/bin/artemis run &
```

The console should have these kinds of message for successfully working.
```
2025-08-04 13:24:51,113 INFO  [org.apache.activemq.artemis.core.server] AMQ221003: Deploying ANYCAST queue ExpiryQueue on address ExpiryQueue
2025-08-04 13:24:51,425 INFO  [org.apache.activemq.artemis.core.server] AMQ221020: Started EPOLL Acceptor at 0.0.0.0:61616 for protocols [CORE,MQTT,AMQP,STOMP,HORNETQ,OPENWIRE]
2025-08-04 13:24:51,426 INFO  [org.apache.activemq.artemis.core.server] AMQ221020: Started EPOLL Acceptor at 0.0.0.0:5445 for protocols [HORNETQ,STOMP]
2025-08-04 13:24:51,428 INFO  [org.apache.activemq.artemis.core.server] AMQ221020: Started EPOLL Acceptor at 0.0.0.0:5672 for protocols [AMQP]
2025-08-04 13:24:51,434 INFO  [org.apache.activemq.artemis.core.server] AMQ221020: Started EPOLL Acceptor at 0.0.0.0:1883 for protocols [MQTT]
2025-08-04 13:24:51,436 INFO  [org.apache.activemq.artemis.core.server] AMQ221020: Started EPOLL Acceptor at 0.0.0.0:61613 for protocols [STOMP]
2025-08-04 13:24:51,439 INFO  [org.apache.activemq.artemis.core.server] AMQ221007: Server is now live
2025-08-04 13:24:51,439 INFO  [org.apache.activemq.artemis.core.server] AMQ221001: Apache ActiveMQ Artemis Message Broker version 2.19.1 [0.0.0.0, nodeID=23572ce5-7160-11f0-bc7d-3cfdfe27b520]
```


Run the baseline software
```bash
./run_baseline.sh input_dir/ output_dir/ output.log
```

**Arguments:**
* 1st is the input directory
* 2nd is the output directory
* 3rd is the path to a log file





The ending message for looks something like this:
```
 2025 07:09:47  INFO PbjSender - Sending processed information to localhost JavaToPy ...


    _/_/_/_/  _/            _/            _/                        _/
   _/            _/_/_/          _/_/_/  _/_/_/      _/_/      _/_/_/
  _/_/_/    _/  _/    _/  _/  _/_/      _/    _/  _/_/_/_/  _/    _/
 _/        _/  _/    _/  _/      _/_/  _/    _/  _/        _/    _/
_/        _/  _/    _/  _/  _/_/_/    _/    _/    _/_/_/    _/_/_/


27 Jul 2025 07:09:47  INFO PbjSender - Sending Stop code to localhost JavaToPy ...
27 Jul 2025 07:09:47  INFO PbjJmsSender - Disconnected PBJ Sender on localhost JavaToPy ...
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Modality filtering turned off, proceeding for patient patient47 note patient47_report054_NOTE
Run Start Time:               Sun Jul 27 07:01:16 CDT 2025
Processing Start Time:        Sun Jul 27 07:01:39 CDT 2025
Processing End Time:          Sun Jul 27 07:09:47 CDT 2025
Initialization Time Elapsed:  23 seconds
Processing Time Elapsed:      8 minutes, 7 seconds
Total Run Time Elapsed:       8 minutes, 31 seconds
Documents Processed:          562
Average Seconds per Document: 0.87
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Modality filtering turned off, proceeding for patient patient47 note patient47_report055_NOTE
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Modality filtering turned off, proceeding for patient patient47 note patient47_report056_NOTE
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Sun Jul 27 07:09:47 2025 Received Stop code.
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Sun Jul 27 07:09:47 2025 Disconnected Stomp Receiver on localhost JavaToPy
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Finished processing notes
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Writing results for all input in ../ovarian_dev_output/unsummarized_output.tsv
27 Jul 2025 07:09:47  INFO timelines_python_pipeline - Finished writing
```

Make sure it says `INFO timelines_python_pipeline - Finished writing`.  This can take a few more seconds of processing after the big "Finished" banner.

The baseline will just stall at this point.  You have to press `Ctrl+C` to kill it.

This produces an `unsummarized_out.tsv` file to the output directory.



## Converting TSV file

This has to be converted to a JSON format for evaluation on the 2025 baseline, but there may be some loss in the data format.

The evaluation code is at https://github.com/HealthNLPorg/chemoTimelinesEval

### Improvements:

It might not be difficult to use `trap` in Bash to capture SIGINT (or Python `signal`), shut down the broker and stop cTAKEs, and start the next step of the process/pipeline.  This is not the most intuitive design.