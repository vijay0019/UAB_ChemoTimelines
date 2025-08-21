import json
import os
import random
import re
import threading
from functools import lru_cache
from pprint import pprint

import dspy
import numpy as np
import pandas as pd
import tiktoken
from typing_extensions import NamedTuple, Literal

from mychatadapter import MyChatAdapter
from threadsafe_ollama import create_threadsafe_models
from config import Config, MODEL, CONTEXT_WINDOW, MIN_TEMPERATURE, MAX_TEMPERATURE
# from dspy import ChatAdapter as MyChatAdapter

import hashlib
import weakref

# Global tokenizer - can be swapped for different model types
_default_tokenizer = tiktoken.get_encoding("cl100k_base")

# Object ID cache for ephemeral objects (avoids serialization cost)
_ephemeral_cache = weakref.WeakKeyDictionary()

def _get_tokenizer():
    """Get the current tokenizer, allowing for runtime swapping."""
    global _default_tokenizer
    return _default_tokenizer

def set_tokenizer(new_tokenizer):
    """Set a new tokenizer and clear both caches to ensure consistency."""
    global _default_tokenizer, _ephemeral_cache
    _default_tokenizer = new_tokenizer
    cached_token_count.cache_clear()
    _ephemeral_cache.clear()

def _serialize_for_tokenization(obj) -> str:
    """Convert object to deterministic string representation."""
    if isinstance(obj, str):
        return obj
    try:
        # Convert NamedTuples and other objects to dict for JSON serialization
        if hasattr(obj, '_asdict'):
            obj = obj._asdict()
        elif hasattr(obj, '__dict__'):
            obj = obj.__dict__
        elif isinstance(obj, (list, tuple)):
            obj = [item._asdict() if hasattr(item, '_asdict') else item for item in obj]
        
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to string representation for non-serializable objects
        return str(obj)

def _hash_text(text: str) -> str:
    """Return SHA1 digest as cache key (fixed-size)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

@lru_cache(maxsize=2000)
def cached_token_count(hash_key: str, serialized_text: str) -> int:
    """Cached tokenization using hash key instead of raw text."""
    tokenizer = _get_tokenizer()
    
    # Handle different tokenizer types
    if hasattr(tokenizer, 'encode'):
        # tiktoken or similar interface
        return len(tokenizer.encode(serialized_text))
    elif hasattr(tokenizer, 'tokenize'):
        # Hugging Face tokenizer interface
        return len(tokenizer.tokenize(serialized_text))
    elif callable(tokenizer):
        # Custom tokenizer function
        result = tokenizer(serialized_text)
        return len(result) if hasattr(result, '__len__') else result
    else:
        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

def get_token_count(obj) -> int:
    """Get token count with optimized caching strategy."""
    # Fast path: check ephemeral cache for objects that can be weakly referenced
    try:
        if obj in _ephemeral_cache:
            return _ephemeral_cache[obj]
    except TypeError:
        # Object is not hashable or weak-referenceable, skip ephemeral cache
        pass
    
    # Stable path: serialize and hash for cache key
    serialized = _serialize_for_tokenization(obj)
    hash_key = _hash_text(serialized)
    result = cached_token_count(hash_key, serialized)
    
    # Store in ephemeral cache if possible (for repeated access to same object)
    try:
        _ephemeral_cache[obj] = result
    except TypeError:
        # Object is not weak-referenceable, skip ephemeral cache
        pass
    
    return result

# Use Config values for task2-specific settings
MAX_RETRIES = 2  # Override for task2
DEFAULT_REPEAT_PENALTY = Config.DEFAULT_REPEAT_PENALTY
DEFAULT_REPEAT_LAST_N = Config.DEFAULT_REPEAT_LAST_N
LOW_REP_REPEAT_PENALTY = Config.LOW_REP_REPEAT_PENALTY
LOW_REP_REPEAT_LAST_N = Config.LOW_REP_REPEAT_LAST_N


pd.set_option('display.max_columns', None)

# Create thread-safe models with default repeat penalty
MODELS = create_threadsafe_models(
    model_name=MODEL,
    context_window=CONTEXT_WINDOW,
    temperature_range=(MIN_TEMPERATURE, MAX_TEMPERATURE),
    num_models=MAX_RETRIES + 1,
    repeat_penalty=DEFAULT_REPEAT_PENALTY,
    repeat_last_n=DEFAULT_REPEAT_LAST_N
)

# Create thread-safe models with low repetition penalty
LOW_REP_MODELS = create_threadsafe_models(
    model_name=MODEL,
    context_window=CONTEXT_WINDOW,
    temperature_range=(MIN_TEMPERATURE, MAX_TEMPERATURE),
    num_models=MAX_RETRIES + 1,
    repeat_penalty=LOW_REP_REPEAT_PENALTY,
    repeat_last_n=LOW_REP_REPEAT_LAST_N
)

dspy.configure(lm=MODELS[0], adapter=MyChatAdapter())

CHEMO_DRUGS = [
    "a.c",
    "a/c",
    "abraxane",
    "ac",
    "adriamycin",
    "aflibercept",
    "alfa-2b interferon",
    "alibercept",
    "alpha-2b interferon",
    "arimidex",
    "avastin",
    "bevacizumab",
    "caboplatin",
    "cabotaxol",
    "carbo",
    "carboplatin",
    "carbotaxol",
    "chemo",
    "chemotherapy",
    "cisplatin",
    "cistoplatin",
    "cyclophosphamide",
    "cytoxan",
    "docetaxel",
    "docetaxol",
    "doxil",
    "doxorubicin",
    "gemcitabine",
    "gemzar",
    "herceptin",
    "il-2",
    "il2",
    "interferon",
    "interleukin-2",
    "ipilimumab",
    "liposomal doxorubicin",
    "methotrexate",
    "paclitaxel",
    "tamoxifen",
    "tax",
    "taxol",
    "taxotere",
    "tc",
    "tch",
    "temozolomide",
    "vaccinia",
    "vaccinia virus"
]

Relation = Literal[
    "begins-on",
    "ends-on",
    "contains-1"
]


class Date(NamedTuple):
	year: int
	month: int|None
	day_of_month: int|None
	week: int|None


def convert_date_to_string(date: Date) -> str:
    """Convert Date tuple to string in competition format."""
    assert date.year is not None
    if date.week is None or date.day_of_month is not None:
        if date.month is None:
            return f"{date.year}"
        if date.day_of_month is None:
            return f"{date.year}-{date.month:02d}"
        return f"{date.year}-{date.month:02d}-{date.day_of_month:02d}"
    return f"{date.year}-w{date.week:02d}"
    


# Type alias for timeline entries
# Using str for drug names instead of Literal[*CHEMO_DRUGS] because:
# 1. Literal doesn't support dynamic expansion of lists
# 2. The model may discover additional drug names beyond the predefined list
TIMELINE = list[tuple[str, Relation, Date]]

EXAMPLE_TIMELINE = """
[
    ('tc', 'contains-1', Date(year=2011, month=8, day_of_month=None, week=None)),
    ('cyclophosphamide', 'begins-on', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('docetaxel', 'begins-on', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('chemo', 'contains-1', Date(year=2011, month=8, day_of_month=10, week=None)),
    ('chemotherapy', 'contains-1', Date(year=2011, month=8, day_of_month=10, week=None)),
    ('docetaxol', 'contains-1', Date(year=2011, month=None, day_of_month=None, week=36)),
    ('cyclophosphamide', 'ends-on', Date(year=2011, month=10, day_of_month=10, week=None)),
    ('chemotherapy', 'ends-on', Date(year=2012, month=None, day_of_month=None, week=None))
]
"""


class ChemoNotesTimeline(dspy.Signature):
    __doc__ = """
Extract chemotherapy events and dates from clinical text.
Exclude surgical procedures, radiation therapy, and other non-chemotherapy-related events.
Use standardized drug names (e.g., 'cyclophosphamide' instead of 'Cytoxan').
Include ALL mentions from the following list (and any additional mentions found in the text):\n""" + "\n".join(sorted(CHEMO_DRUGS))
    Notes: str = dspy.InputField(desc="clinical text")
    Timeline: str = dspy.OutputField(desc="structured events")


class ChemoTimelineUpdate(dspy.Signature):
    __doc__ = """
Extract therapies and temporal relations from clinical text and return as structured tuples: (therapy, relation, date)
Exclude surgical procedures, radiation therapy, and other non-chemotherapy-related events.

Therapies: Use generic drug names (cyclophosphamide, docetaxel, chemotherapy, etc.)

Relations:
- 'begins-on': treatment/medication starts
- 'ends-on': treatment/medication ends
- 'contains-1': treatment occurred within timeframe

Acceptable date formats (in order of preference):
1. Specify year, month, and day.
2. Specify year and week.
3. Specify year and month.
4. Specify year only.

Example output format:
[[ ## timeline_update ## ]]""" + EXAMPLE_TIMELINE + """[[ ## completed ## ]]
    """
    timeline: TIMELINE = dspy.InputField(desc="existing events")
    chunk_content: str = dspy.InputField(desc="current text chunk")
    timeline_update: TIMELINE = dspy.OutputField(desc="new events")


class ChemoTimelineCleanup(dspy.Signature):
    __doc__ = """
Consolidate and clean up events. Remove duplicates and resolve conflicts.
Maintain the tuple format: (entity, relation, date)
Sort by date and ensure logical consistency (begins-on before ends-on for same entity).

Example output format:
[[ ## cleaned_timeline ## ]]""" + EXAMPLE_TIMELINE + """[[ ## completed ## ]]
    """
    timeline: TIMELINE = dspy.InputField()
    cleaned_timeline: TIMELINE = dspy.OutputField()


class ChemoTimelineBuilder(dspy.Module):
    def __init__(self, starting_chunks: int = 1, intermediate_chunks: int = 1,
                 token_threshold: int = CONTEXT_WINDOW * 0.25, timeline_cleanup_threshold: int = 10):
        super().__init__()
        self.debrief_lm = dspy.ChainOfThought(ChemoNotesTimeline)
        self.update_lm = dspy.ChainOfThought(ChemoTimelineUpdate)
        self.cleanup_lm = dspy.ChainOfThought(ChemoTimelineCleanup)
        self.starting_chunks = starting_chunks
        self.intermediate_chunks = intermediate_chunks
        self.token_threshold = token_threshold
        self.timeline_cleanup_threshold = timeline_cleanup_threshold

    def retry(self, func, high_rep_penalty=False, **kwargs):
        for model in MODELS if not high_rep_penalty else LOW_REP_MODELS:
            with dspy.context(lm=model, adapter=MyChatAdapter()):
                output = func(**kwargs)

            if not any(x is None for x in output.values()):
                return output

            # Handle specific None cases
            if "Timeline" in output and output["Timeline"] is None:
                output["Timeline"] = kwargs.get("Notes", "")
                return output
            if "timeline_update" in output and output["timeline_update"] is None:
                output["timeline_update"] = []
                return output
            if "cleaned_timeline" in output and output["cleaned_timeline"] is None:
                output["cleaned_timeline"] = kwargs.get("timeline", [])
                return output

        raise RuntimeError(f"Failed to get valid response after {MAX_RETRIES} retries.")

    def process_timeline_update(self, current_timeline, new_events):
        """Merge new events with current timeline"""
        if not new_events:
            return current_timeline

        # Combine with existing timeline
        all_events = list(current_timeline) + new_events

        # Remove duplicates while preserving order
        seen = set()
        unique_events = []
        for event in all_events:
            if event not in seen:
                seen.add(event)
                unique_events.append(event)

        return sorted(unique_events, key=lambda x: self._get_sort_key(x))

    def _get_sort_key(self, event):
        """Get sort key for timeline event, handling both Date objects and string dates"""
        drug, relation, date = event
        
        # Handle Date objects
        if hasattr(date, 'year'):
            return (date.year, date.month or 0, date.day_of_month or 0, date.week or 0, drug, relation)
        
        # Handle string dates - parse them for sorting
        date_str = str(date)
        
        # Parse different date formats for sorting
        # Format: YYYY-MM-DD, YYYY-MM, YYYY, YYYY-wWW
        parts = date_str.split('-')
        
        try:
            year = int(parts[0])
            
            if len(parts) == 1:
                # Just year: YYYY
                return (year, 0, 0, 0, drug, relation)
            elif len(parts) == 2:
                if parts[1].startswith('w'):
                    # Week format: YYYY-wWW
                    week = int(parts[1][1:])
                    return (year, 0, 0, week, drug, relation)
                else:
                    # Month format: YYYY-MM
                    month = int(parts[1])
                    return (year, month, 0, 0, drug, relation)
            elif len(parts) == 3:
                # Full date: YYYY-MM-DD
                month = int(parts[1])
                day = int(parts[2])
                return (year, month, day, 0, drug, relation)
        except (ValueError, IndexError):
            # Fallback for unparseable dates - sort by string
            pass
        
        # Fallback: sort by string representation
        return (9999, 99, 99, 99, drug, relation)  # Put unparseable dates at the end

    def evaluate_and_update_timeline(self, timeline, content, reasoning):
        """Process content and update timeline"""
        # Check if content is too long
        if get_token_count(content) > self.token_threshold:
            output = self.retry(self.debrief_lm, Notes=content)
            if output.get("reasoning"):
                reasoning.append(output["reasoning"])
            content = output["Timeline"]

        # Update timeline with new content
        output = self.retry(self.update_lm, timeline=timeline, chunk_content=content)
        if output.get("reasoning"):
            reasoning.append(output["reasoning"])

        # Process the update
        timeline = self.process_timeline_update(timeline, output.get("timeline_update", []))

        # Clean up if timeline is getting long
        if len(timeline) > self.timeline_cleanup_threshold:
            output = self.retry(self.cleanup_lm, timeline=timeline, high_rep_penalty=True)
            if output.get("reasoning"):
                reasoning.append(output["reasoning"])
            timeline = output.get("cleaned_timeline", timeline)

        return timeline

    def build_generic_timeline(self, chunks, timeline, reasoning):
        """Build timeline from text chunks"""
        # Process initial chunks
        content = "\n".join(chunks[:self.starting_chunks])
        timeline = self.evaluate_and_update_timeline(timeline, content, reasoning)

        # Process remaining chunks
        for i in range(self.starting_chunks, len(chunks), self.intermediate_chunks):
            chunk_set = chunks[i:i + self.intermediate_chunks]
            content = "\n".join(chunk_set)
            timeline = self.evaluate_and_update_timeline(timeline, content, reasoning)

        # Convert dates to strings
        timeline = [(drug, relation, convert_date_to_string(date)) for drug, relation, date in timeline]

        # Sort using custom key that handles string dates properly
        return sorted(timeline, key=lambda x: self._get_sort_key(x))

    def forward(self, chunks: list[str]):
        # Initialize
        timeline = []
        reasoning = []

        # Build timeline from chunks
        timeline = self.build_generic_timeline(chunks, timeline, reasoning)

        return dspy.Prediction(
            timeline=timeline,
            reasoning="\n\n\n".join(reasoning)
        )


def get_prompt_optimizer(optimizer_name: str, metric, **kwargs):
    """Get a DSPy optimizer based on the name."""
    if optimizer_name == 'simba':
        return dspy.SIMBA(
            metric=metric,
            bsize=kwargs.get('bsize', Config.SIMBA_BSIZE),
            num_candidates=kwargs.get('num_candidates', Config.SIMBA_NUM_CANDIDATES),
            max_steps=kwargs.get('max_steps', Config.SIMBA_MAX_STEPS),
            num_threads=kwargs.get('num_threads', Config.SIMBA_NUM_THREADS)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: simba")

def evaluate(train, dev, zeroshot, optimize=None):
    def timeline_f1(example, pred, trace=None):
        """Simplified metric for timeline comparison"""
        if not pred.timeline and not example.timeline:
            return 1.0
        if not pred.timeline or not example.timeline:
            return 0.0

        # Convert to sets for easier comparison
        pred_set = set(pred.timeline)
        true_set = set(example.timeline)

        # Calculate precision and recall
        true_positives = len(pred_set & true_set)
        precision = true_positives / len(pred_set)
        recall = true_positives / len(true_set)

        # F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # # Split train into train and validation sets
    # val = [example for example in train if sum([len(tokenizer.encode(chunk)) for chunk in example.chunks]) >= CONTEXT_WINDOW or len(example.timeline) == 0]
    # train = [example for example in train if sum([len(tokenizer.encode(chunk)) for chunk in example.chunks]) < CONTEXT_WINDOW and len(example.timeline) > 0]
    # print(f"Train examples: {len(train)}, Validation examples: {len(val)}")

    # Define evaluator
    evaluator = dspy.Evaluate(devset=dev,
                              metric=timeline_f1,
                              num_threads=16,
                              display_progress=True,
                              max_errors=0)

    # Evaluate zero-shot model
    print("Zero-shot results:")
    evaluation = evaluator(zeroshot)
    acc_zero, outputs_zero = evaluation["score"], evaluation["results"]
    print(f"Zero-shot accuracy: {acc_zero}")
    
    # Use configuration-based optimization setting if not explicitly provided
    if optimize is None:
        optimize = Config.ENABLE_PROMPT_OPTIMIZATION
        
    if not optimize:
        print("⚠️  WARNING: Running without prompt optimization!")
        print("   This may result in lower accuracy. To enable optimization:")
        print("   export CHEMO_PROMPT_OPTIMIZER=simba")
        print("Skipping optimization.")
        return acc_zero, outputs_zero, zeroshot

    # Train few-shot model
    optimizer = get_prompt_optimizer(Config.PROMPT_OPTIMIZER, metric=timeline_f1, num_threads=16)
    fewshot = optimizer.compile(zeroshot, trainset=train)

    # Evaluate few-shot model
    print("Few-shot results:")
    evaluation = evaluator(fewshot)
    acc_few, outputs_few = evaluation["score"], evaluation["results"]
    print(f"Few-shot accuracy: {acc_few}")
    
    # print(dspy.inspect_history(10))

    if acc_zero < acc_few:
        fewshot.save("fewshot_model.json")
        print("Saved improved few-shot model")
        return acc_few, outputs_few, fewshot
    else:
        print("Few-shot model did not improve over zero-shot model")
        return acc_zero, outputs_zero, zeroshot
        

def concatenate_chunks(data, target):
    data = data.copy()
    # Concatenate chunks intelligently to fit within the context window
    for split in data:
        # Avoid creating unnecessary list from items()
        for key in list(data[split]["chunks"].keys()):
            value = data[split]["chunks"][key]
            # Concatenate chunks if they fit within the context window
            concat = "\n\n\n".join(v for k, v in value)
            n = get_token_count(concat)
            if len(value) == 1 or n < target:
                data[split]["chunks"][key] = [concat]
            # If more than one chunk, concatenate them intelligently
            else:
                # Group chunks by report ID and concatenate them
                chunks = ["\n\n\n".join(v for k, v in value if k == i) for i in sorted(set(k for k, v in value))]
                # Concatenate chunks while respecting the context window
                # Use a greedy approach to concatenate as many chunks as possible without exceeding the context window
                # This is a simple heuristic and may not be optimal
                concatenated_chunks = []
                current_chunk = chunks[0]
                # target = math.ceil(n / math.ceil(2 * n / CONTEXT_WINDOW))
                for i, chunk in enumerate(chunks[1:], 1):
                    # Check if adding the next chunk exceeds the context window
                    test_chunk = current_chunk + "\n\n\n" + chunk
                    if get_token_count(test_chunk) < target:
                        current_chunk = test_chunk
                    else:
                        concatenated_chunks.append(current_chunk)
                        current_chunk = chunk
                if current_chunk:
                    concatenated_chunks.append(current_chunk)
                data[split]["chunks"][key] = concatenated_chunks

    # Rearrange data structure - optimize by pre-caching timeline keys
    for split in data:
        new_data = {}
        timeline_keys = set(data[split]["timeline"].keys())  # Pre-compute set for faster lookups
        for patient, chunks in data[split]["chunks"].items():
            if patient in timeline_keys:
                new_data[patient] = dspy.Example({
                    "chunks": chunks,
                    "timeline": data[split]["timeline"][patient]
                }).with_inputs("chunks")
        data[split] = new_data
        
    return data


if __name__ == "__main__":
    import argparse
    import json
    import math
    import os
    from collections import defaultdict
    from copy import deepcopy
    from functools import partial
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Run Task2 timeline generation')
    parser.add_argument('--output-dir', default='.', help='Output directory for generated timelines (default: current directory)')
    args = parser.parse_args()

    data = {split: {"chunks": defaultdict(list), "timeline": {}} for split in ["train", "dev"]}
    notes_path = "chemoTimelines2024_train_dev_labeled/subtask1/Patient_Notes"
    timelines_path = "chemoTimelines2024_train_dev_labeled/subtask1/Gold_Timelines_allPatients_processed"

    # Load timelines and notes
    report_type = lambda x: x.split('_')[-1].strip(".txt").upper()
    for file in os.listdir(timelines_path):
        if file.endswith(".json"):
            split, site, patient = file[:-5].split("_")
            with open(os.path.join(timelines_path, file), 'r') as f:
                timelines = json.load(f)
            data[split]["timeline"][f"{site}_{patient}"] = [tuple(item) for item in timelines]
            reports_path = os.path.join(notes_path, site, split, patient)
            for report in sorted(os.listdir(reports_path), key=lambda x: ['NOTE', 'PGN', 'RAD', report_type(x)].index(report_type(x))):
                with open(os.path.join(reports_path, report), 'r') as f:
                    content = f.read().strip()
                data[split]["chunks"][f"{site}_{patient}"].append((re.search(r'report(\d+)', report).group(1), content))

    data_old = deepcopy(data)
    data = concatenate_chunks(data, target=CONTEXT_WINDOW * 0.75)

    train_data = list(data["train"].values())
    dev_data = list(data["dev"].values())

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")

    # Warn user about optimization status
    Config.warn_if_optimization_disabled()

    # acc_large, outputs_large, zeroshot = evaluate(train_data, dev_data, ChemoTimelineBuilder(), optimize=False)
    # data = concatenate_chunks(data_old, target=CONTEXT_WINDOW * 0.25)
    # train_data = list(data["train"].values())
    # dev_data = list(data["dev"].values())
    # acc_small, outputs_small, fewshot = evaluate(train_data, dev_data, ChemoTimelineBuilder())
    # if acc_small > acc_large:
    #     print(f"Small context window model ({acc_small}) outperformed large context window model ({acc_large}).")
    #     builder = zeroshot
    # else:
    #     print(f"Large context window model ({acc_large}) outperformed small context window model ({acc_small}).")
    #     builder = fewshot
    
    _, _, builder = evaluate(train_data, dev_data, ChemoTimelineBuilder())

    jsons = defaultdict(dict)
    for split in ["train", "dev"]:
        for key, value in tqdm(data[split].items(), desc=f"Processing {split} data"):
            site, patient = key.split('_')
            generated = builder(value["chunks"])
            for entry in list(generated.timeline):
                if not re.match(r'^\d{4}(?:-(?:\d{2}(?:-\d{2})?|w\d{2}))?$', entry[2]):
                    print(f"Invalid date format in entry {entry} for {patient} in {site} {split}. Removing entry.")
                    generated.timeline.remove(entry)
            jsons[f"{site}_{split}"][patient] = generated.timeline
    
    # Create model postfix by cleaning up the model name
    model_postfix = MODEL.replace('/', '_').replace(':', '_')
    task_postfix = "task2"
    
    for site_split, timelines in jsons.items():
        filename = f"{site_split}_all_patients_generated_timelines_{model_postfix}_{task_postfix}.json"
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(timelines, f, indent=2)
            
    print("Timeline examples created successfully.")
