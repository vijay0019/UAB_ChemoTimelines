import json
import os
import random
import re
import threading
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

tokenizer = tiktoken.get_encoding("cl100k_base")

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
    


TIMELINE = list[tuple[Literal[*CHEMO_DRUGS], Relation, Date]]

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

        return sorted(unique_events, key=lambda x: (x[2].year, x[2].month or 0, x[2].day_of_month or 0, x[2].week or 0, x[0], x[1]))  # Sort by date, then by drug and relation

    def evaluate_and_update_timeline(self, timeline, content, reasoning):
        """Process content and update timeline"""
        # Check if content is too long
        if len(tokenizer.encode(content)) > self.token_threshold:
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

        return sorted(timeline)

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


def evaluate(train, dev, zeroshot, optimize=True):
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
                              return_outputs=True,
                              max_errors=0)

    # Evaluate zero-shot model
    print("Zero-shot results:")
    evaluation = evaluator(zeroshot)
    acc_zero, outputs_zero = evaluation["score"], evaluation["results"]
    print(f"Zero-shot accuracy: {acc_zero}")
    
    if not optimize:
        print("Skipping optimization.")
        return acc_zero, outputs_zero, zeroshot

    # Train few-shot model
    optimizer = dspy.SIMBA(metric=timeline_f1, num_threads=16)
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
        items = list(data[split]["chunks"].items())
        for key, value in items:
            # Concatenate chunks if they fit within the context window
            concat = "\n\n\n".join(v for k, v in value)
            n = len(tokenizer.encode(concat))
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
                    if len(tokenizer.encode(current_chunk + "\n\n\n" + chunk)) < target:
                        current_chunk += "\n\n\n" + chunk
                    else:
                        concatenated_chunks.append(current_chunk)
                        current_chunk = chunk
                if current_chunk:
                    concatenated_chunks.append(current_chunk)
                data[split]["chunks"][key] = concatenated_chunks

    # Rearrange data structure
    for split in data:
        new_data = {}
        for patient, chunks in data[split]["chunks"].items():
            if patient in data[split]["timeline"]:
                new_data[patient] = dspy.Example({
                    "chunks": chunks,
                    "timeline": data[split]["timeline"][patient]
                }).with_inputs("chunks")
        data[split] = new_data
        
    return data


if __name__ == "__main__":
    import json
    import math
    import os
    from collections import defaultdict
    from copy import deepcopy
    from functools import partial
    from tqdm import tqdm

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
    
    _, _, builder = evaluate(train_data, dev_data, ChemoTimelineBuilder(), optimize=False)

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
    for site_split, timelines in jsons.items():
        with open(f"{site_split}_all_patients_generated_timelines.json", "w") as f:
            json.dump(timelines, f, indent=2)
            
    print("Timeline examples created successfully.")
