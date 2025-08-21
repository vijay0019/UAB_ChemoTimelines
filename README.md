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

### Prompt Optimization Configuration
- `CHEMO_ENABLE_PROMPT_OPTIMIZATION`: Enable prompt optimization (default: `false`)
- `CHEMO_PROMPT_OPTIMIZER`: Optimizer type (default: `simba`)
- `CHEMO_SIMBA_BSIZE`: SIMBA batch size (default: `4`)
- `CHEMO_SIMBA_NUM_CANDIDATES`: Number of candidates (default: `10`)
- `CHEMO_SIMBA_MAX_STEPS`: Maximum optimization steps (default: `3`)
- `CHEMO_SIMBA_NUM_THREADS`: SIMBA threads (default: `1`)

Example configuration:
```bash
export CHEMO_MODEL="ollama/qwen3:30b"
export CHEMO_CONTEXT_WINDOW="65536"
export OLLAMA_PORTS="11435,11436,11437,11438"
export CHEMO_ENABLE_PROMPT_OPTIMIZATION="true"
export CHEMO_PROMPT_OPTIMIZER="simba"
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

#### Standard Version
```bash
cd timeline_generation
python task2.py
```

#### Enhanced Version with TLINK Integration
```bash
cd timeline_generation
# With TLINK prediction (requires trained relation model)
python task2_with_tlinks.py --tlink-model-dir /path/to/trained/model

# Without TLINK prediction (fallback to standard behavior)
python task2_with_tlinks.py --disable-tlinks

# Using pre-computed entity predictions from file
python task2_with_tlinks.py --dev-entities /path/to/entities.csv
```

**Command Line Options for Enhanced Version:**
- `--tlink-model-dir`: Path to trained temporal relation model
- `--disable-tlinks`: Disable TLINK prediction 
- `--dev-entities`: Path to CSV file with predicted entities (default: `data/dev/dev_entities_events_subtask2.csv`)
- `--output-dir`: Directory for generated timeline files

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

## TLINK Integration

The enhanced Task 2 (`task2_with_tlinks.py`) includes temporal link (TLINK) prediction to improve timeline generation accuracy.

### Features

- **Temporal Relation Prediction**: Uses a trained BERT-based model to predict temporal relations between events and time expressions
- **Entity Recognition**: Specialized extraction for chemotherapy-related entities
- **Pre-computed Entities**: Support for using pre-computed entity predictions from CSV files
- **Fallback Support**: Graceful degradation when TLINK models are unavailable
- **Bug Fixes**: Includes critical fixes for timeline sorting and type safety

### Entity File Format

When using `--dev-entities`, the CSV file should contain columns:
- `text`: Entity text
- `start`: Character start position
- `end`: Character end position  
- `entity_type`: Type of entity (EVENT, TREATMENT, MEDICATION, or TIMEX3)

### Dependencies for TLINK Features

```bash
pip install torch transformers scikit-learn spacy medspacy
```

## Recent Improvements

### Bug Fixes
- **Timeline Sorting**: Fixed AttributeError when sorting mixed Date objects and strings
- **Chronological Ordering**: Replaced lexicographic sorting with proper chronological sorting
- **Type Safety**: Fixed invalid Literal type definitions

### Performance Optimizations
- **Token Caching**: Implemented efficient token counting with dual-cache system
- **Memory Management**: Optimized data structures and reduced memory usage
- **Threading Safety**: Thread-safe model management for concurrent processing

### Documentation
- Comprehensive integration documentation in `TLINK_INTEGRATION_README.md`
- Bug fix summary in `timeline_generation/BUGFIX_SUMMARY.md`
- Test coverage for all major components

For detailed information about TLINK integration, see `TLINK_INTEGRATION_README.md`.
