# ChemoTimelines

Clinical chemotherapy timeline extraction and analysis system that processes patient medical records to automatically generate structured timelines of chemotherapy treatments.

## Overview

ChemoTimelines provides two main tasks:

- **Task 1**: SACT timeline building from patient reports with temporal relations
- **Task 2**: Direct chemotherapy timeline extraction from clinical text chunks

Both tasks use large language models through Ollama to process clinical text and extract structured treatment timelines.

## Requirements

### Ollama Setup
This project requires Ollama to be installed and running. Install Ollama from [https://ollama.ai](https://ollama.ai).

Make sure Ollama is running on the expected ports (default: 11435-11438):
```bash
ollama serve --port 11435 &
ollama serve --port 11436 &
ollama serve --port 11437 &
ollama serve --port 11438 &
```

### Python Dependencies
Install required Python packages:
```bash
pip install dspy-ai tiktoken xmltodict tqdm typing-extensions
```

## Configuration

The system uses environment variables for configuration. Key variables include:

### Model Configuration
- `CHEMO_MODEL`: Model name (default: `ollama/phi4:latest`)
- `CHEMO_CONTEXT_WINDOW`: Context window size (default: `16384`)
- `CHEMO_MIN_TEMPERATURE`: Minimum temperature (default: `0.2`)
- `CHEMO_MAX_TEMPERATURE`: Maximum temperature (default: `1.0`)
- `CHEMO_MAX_RETRIES`: Maximum retries (default: `8`)

### Ollama Configuration
- `OLLAMA_PORTS`: Comma-separated list of Ollama ports (default: `11435,11436,11437,11438`)

### Threading Configuration
- `CHEMO_NUM_THREADS`: Number of threads (default: `1`)
- `CHEMO_ENABLE_THREADING`: Enable threading (default: `true`)

Example configuration:
```bash
export CHEMO_MODEL="ollama/qwen3:30b"
export CHEMO_CONTEXT_WINDOW="65536"
export OLLAMA_PORTS="11435,11436,11437,11438"
```

## Running Tasks

### Task 1: SACT Timeline Building

Task 1 processes patient reports with temporal relations to build SACT (Systemic Anti-Cancer Therapy) timelines.

```bash
cd timeline_generation
python task1.py
```

**Required data structure:**
- Patient notes in `chemoTimelines2024_train_dev_labeled/subtask1/Patient_Notes/`
- XML temporal annotations in `chemoTimelines2024_train_dev_labeled/subtask1/Gold_PairWise_Annotations/`
- Gold timelines in `chemoTimelines2024_train_dev_labeled/subtask1/Gold_Timelines_allPatients_processed/`

**Output:** Generated timeline JSON files for each site/split combination.

### Task 2: Direct Timeline Extraction

Task 2 directly extracts chemotherapy timelines from clinical text chunks.

```bash
cd timeline_generation
python task2.py
```

**Required data structure:**
- Patient notes in `chemoTimelines2024_train_dev_labeled/subtask1/Patient_Notes/`
- Gold timelines in `chemoTimelines2024_train_dev_labeled/subtask1/Gold_Timelines_allPatients_processed/`

**Output:** Generated timeline JSON files for each site/split combination.

## Data Paths

Both tasks expect the training data to be located at:
```
chemoTimelines2024_train_dev_labeled/subtask1/
├── Patient_Notes/
├── Gold_PairWise_Annotations/ (Task 1 only)
└── Gold_Timelines_allPatients_processed/
```

Update the data paths in the scripts if your data is located elsewhere.
