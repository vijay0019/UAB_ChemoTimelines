import json
import os
import random
import re
import threading
from datetime import date, timedelta
from pprint import pprint
from time import sleep

import dspy
import numpy as np
import pandas as pd
import pynvml
import tiktoken
from typing_extensions import NamedTuple, Literal

from mychatadapter import MyChatAdapter
# from dspy import ChatAdapter as MyChatAdapter

tokenizer = tiktoken.get_encoding("cl100k_base")

MODEL = "ollama/phi4:latest"
CONTEXT_WINDOW = 16384
MIN_TEMPERATURE = 0.2
MAX_TEMPERATURE = 1.0
MAX_RETRIES = 2
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_REPEAT_LAST_N = 64
LOW_REP_REPEAT_PENALTY = 1.25
LOW_REP_REPEAT_LAST_N = 160


class ThreadSafeOllamaLM(dspy.LM):
    def __init__(self, ports=(11435, 11436, 11437), **kwargs):
        self.lms = []
        self._counter = random.randint(0, len(ports) - 1)
        self._lock = threading.Lock()
        self.kwargs = kwargs

        for port in ports:
            self.lms.append(dspy.LM(base_url=f"http://127.0.0.1:{port}", **kwargs))

        self.model = kwargs.pop("model")

    @staticmethod
    def get_gpu_utilization():
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(deviceCount)]
        utils = [pynvml.nvmlDeviceGetUtilizationRates(handle) for handle in handles]
        mems = [pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in handles]
        return utils, mems

    def _get_next_gpu(self):
        sleep(random.random() * 0.1)  # small delay to avoid contention
        utils, mems = self.get_gpu_utilization()
        least_utilized = min(range(deviceCount), key=lambda i: (utils[i].gpu, mems[i].used))
        if random.random() < 0.5 or utils[least_utilized].gpu == 0:
            with self._lock:
                self._counter = (self._counter + 1) % len(self.lms)
            return self._counter
        return least_utilized

    def __call__(self, **kwargs):
        lm_idx = self._get_next_gpu()
        return self.lms[lm_idx](**kwargs)

    def generate(self, **kwargs):
        lm_idx = self._get_next_gpu()
        return self.lms[lm_idx].generate(**kwargs)


pd.set_option('display.max_columns', None)

pynvml.nvmlInit()
deviceCount = pynvml.nvmlDeviceGetCount()


def get_temperature(retry_count):
    return MIN_TEMPERATURE + (
        (MAX_TEMPERATURE - MIN_TEMPERATURE) * retry_count / MAX_RETRIES if MAX_RETRIES > 0 else 0.0)


MODEL_KWARGS = {"model": MODEL,
                "max_tokens": CONTEXT_WINDOW,
                "num_ctx": CONTEXT_WINDOW,
                "repeat_penalty": DEFAULT_REPEAT_PENALTY,
                "repeat_last_n": DEFAULT_REPEAT_LAST_N}
MODELS = []
for i in range(MAX_RETRIES + 1):
    temperature = get_temperature(i)
    MODEL_KWARGS["temperature"] = temperature
    MODEL_KWARGS["seed"] = i
    MODELS.append(ThreadSafeOllamaLM(**MODEL_KWARGS))

LOW_REP_MODELS = []
MODEL_KWARGS["repeat_penalty"] = LOW_REP_REPEAT_PENALTY  # higher penalty for low repetition models
MODEL_KWARGS["repeat_last_n"] = LOW_REP_REPEAT_LAST_N  # higher penalty for low repetition models
for i in range(MAX_RETRIES + 1):
    temperature = get_temperature(i)
    MODEL_KWARGS["temperature"] = temperature
    MODEL_KWARGS["seed"] = i
    LOW_REP_MODELS.append(ThreadSafeOllamaLM(**MODEL_KWARGS))

dspy.configure(lm=MODELS[0], adapter=MyChatAdapter())


class ChemoNotesTimeline(dspy.Signature):
    """
Extract chemotherapy events and dates from clinical text.
Exclude surgical procedures, radiation therapy, and other non-chemotherapy-related events.
Use standardized drug names (e.g., 'cyclophosphamide' instead of 'Cytoxan').
    """
    Notes: str = dspy.InputField(desc="clinical text")
    Timeline: str = dspy.OutputField(desc="structured events")


class ChemoTimelineUpdate(dspy.Signature):
    """
Extract therapies and temporal relations from clinical text and return as structured tuples: (therapy, relation, date_string)
Exclude surgical procedures, radiation therapy, and other non-chemotherapy-related events.

Therapies: Use generic drug names (cyclophosphamide, docetaxel, chemotherapy, etc.)
Relations:
- 'begins-on': treatment/medication starts
- 'ends-on': treatment/medication ends
- 'contains-1': treatment occurred within timeframe

Date formats:
- Exact dates: 'YYYY-MM-DD' (e.g., '2011-08-08')
- Month only: 'YYYY-MM' (e.g., '2011-08')
- Week: 'YYYY-wWW' (e.g., '2011-w32')
- Year: 'YYYY' (e.g., '2011')

Example output format:
[[ ## timeline_update ## ]]
[
    ('tc', 'contains-1', '2011-08'),
    ('cyclophosphamide', 'begins-on', '2011-08-08'),
    ('docetaxel', 'begins-on', '2011-08-08'),
    ('chemotherapy', 'contains-1', '2011-08-10'),
    ('cyclophosphamide', 'ends-on', '2011-10-10')
]
[[ ## completed ## ]]
    """
    previous_timeline: list[tuple[str, Literal["begins-on", "ends-on", "contains-1"], str]] = dspy.InputField(desc="existing events")
    chunk_content: str = dspy.InputField(desc="current text chunk")
    timeline_update: list[tuple[str, Literal["begins-on", "ends-on", "contains-1"], str]] = dspy.OutputField(desc="new events")


class ChemoTimelineCleanup(dspy.Signature):
    """
Consolidate and clean up events. Remove duplicates and resolve conflicts.
Maintain the tuple format: (entity, relation, date_string)
Sort by date and ensure logical consistency (begins-on before ends-on for same entity).

Example output format:
[[ ## cleaned_timeline ## ]]
[
    ('tc', 'contains-1', '2011-08'),
    ('cyclophosphamide', 'begins-on', '2011-08-08'),
    ('docetaxel', 'begins-on', '2011-08-08'),
    ('chemotherapy', 'contains-1', '2011-08-10'),
    ('cyclophosphamide', 'ends-on', '2011-10-10')
]
[[ ## completed ## ]]
    """
    timeline: list[tuple[str, Literal["begins-on", "ends-on", "contains-1"], str]] = dspy.InputField()
    cleaned_timeline: list[tuple[str, Literal["begins-on", "ends-on", "contains-1"], str]] = dspy.OutputField()


class ChemoTimelineBuilder(dspy.Module):
    def __init__(self, starting_chunks: int = 1, intermediate_chunks: int = 1,
                 token_threshold: int = 2048, timeline_cleanup_threshold: int = 10):
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

        # Standardize and validate new events
        processed_events = []
        for event in new_events:
            if len(event) == 3:
                entity, relation, date_str = event
                if date_str:
                    processed_events.append((entity, relation, date_str))

        # Combine with existing timeline
        all_events = list(current_timeline) + processed_events

        # Remove duplicates while preserving order
        seen = set()
        unique_events = []
        for event in all_events:
            if event not in seen:
                seen.add(event)
                unique_events.append(event)

        return sorted(unique_events)

    def evaluate_and_update_timeline(self, timeline, content, reasoning):
        """Process content and update timeline"""
        # Check if content is too long
        if len(tokenizer.encode(content)) > self.token_threshold:
            output = self.retry(self.debrief_lm, Notes=content)
            if output.get("reasoning"):
                reasoning.append(output["reasoning"])
            content = output["Timeline"]

        # Update timeline with new content
        output = self.retry(self.update_lm, previous_timeline=timeline, chunk_content=content)
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


def evaluate(train, dev, zeroshot):
    def timeline_metric(example, pred, trace=None):
        """Simple exact match for timeline comparison"""
        try:
            return sorted(example.timeline) == sorted(pred.timeline)
        except:
            return False

    # Define evaluator
    evaluator = dspy.Evaluate(devset=dev,
                              metric=timeline_metric,
                              num_threads=8,
                              display_progress=True,
                              return_outputs=True,
                              max_errors=0)

    # Evaluate zero-shot model
    print("Zero-shot results:")
    acc_zero, outputs_zero = evaluator(zeroshot)
    print(f"Zero-shot accuracy: {acc_zero}")

    # Train few-shot model
    optimizer = dspy.SIMBA(metric=timeline_metric)
    fewshot = optimizer.compile(zeroshot, trainset=train)

    # Evaluate few-shot model
    print("Few-shot results:")
    acc_few, outputs_few = evaluator(fewshot)
    print(f"Few-shot accuracy: {acc_few}")

    if acc_zero < acc_few:
        fewshot.save("fewshot_model.json")
        print("Saved improved few-shot model")

    print(dspy.inspect_history(10))


def make_timeline_example(pair, builder, split):
    """Create a dspy.Example from a patient-chunks pair"""
    key, value = pair
    generated = builder(value["chunks"])
    with open(f"{split}_{key}.json", "w") as f:
        json.dump(generated.timeline, f, indent=2)


if __name__ == "__main__":
    import json
    import os
    from collections import defaultdict
    from functools import partial
    from tqdm import tqdm

    data = {split: {"chunks": defaultdict(list), "timeline": {}} for split in ["train", "dev"]}
    notes_path = "chemoTimelines2024_train_dev_labeled/subtask1/Patient_Notes"
    timelines_path = "chemoTimelines2024_train_dev_labeled/subtask1/Gold_Timelines_allPatients_processed"

    # Load timelines and notes
    for timeline in os.listdir(timelines_path):
        if timeline.endswith(".json"):
            split, site, patient = timeline[:-5].split("_")
            with open(os.path.join(timelines_path, timeline), 'r') as f:
                timelines = json.load(f)
            data[split]["timeline"][f"{site}_{patient}"] = [tuple(item) for item in timelines]
            report_path = os.path.join(notes_path, site, split, patient)
            for report in os.listdir(report_path):
                with open(os.path.join(report_path, report), 'r') as f:
                    content = f.read()
                data[split]["chunks"][f"{site}_{patient}"].append(content)

    # Concatenate chunks for each patient
    for split in data:
        for patient, chunks in data[split]["chunks"].items():
            if len(chunks) > 1:
                concatenated_chunks = []
                current_chunk = chunks[0].strip()
                for chunk in chunks[1:]:
                    if len(tokenizer.encode(current_chunk + "\n\n\n" + chunk.strip())) < 8192:
                        current_chunk += "\n\n\n" + chunk.strip()
                    else:
                        concatenated_chunks.append(current_chunk.strip())
                        current_chunk = chunk.strip()
                if current_chunk:
                    concatenated_chunks.append(current_chunk.strip())
                data[split]["chunks"][patient] = concatenated_chunks

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

    train_data = list(data["train"].values())
    dev_data = list(data["dev"].values())

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")

    # evaluate(train_data, dev_data, ChemoTimelineBuilder())
    
    builder = ChemoTimelineBuilder()
    for split in ["train", "dev"]:
        for pair in tqdm(data[split].items(), desc=f"Processing {split} data"):
            make_timeline_example(pair, builder, split)
    print("Timeline examples created successfully.")
