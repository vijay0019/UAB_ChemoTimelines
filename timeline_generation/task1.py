import json
import os
import random
import re
import threading
from pprint import pprint
from time import sleep

import dspy
import numpy as np
import pandas as pd
import pynvml
import tiktoken
import xmltodict
from typing_extensions import NamedTuple, Literal

from mychatadapter import MyChatAdapter
# from dspy import ChatAdapter as MyChatAdapter

tokenizer = tiktoken.get_encoding("cl100k_base")

MODEL = "ollama/phi4:latest"
CONTEXT_WINDOW = 16384
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 0.5
MAX_RETRIES = 5


class ThreadSafeOllamaLM(dspy.LM):
    def __init__(self, ports=(11435, 11436, 11437, 11438), **kwargs):
        self.lms = []
        # self._counter = random.randint(0, len(ports) - 1)
        # self._lock = threading.Lock()
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
        sleep(random.random() * 0.1 + 0.1)  # small delay to avoid contention
        utils, mems = self.get_gpu_utilization()
        least_utilized = min(range(deviceCount), key=lambda i: (utils[i].gpu, mems[i].used, random.random()))
        # if random.random() < 0.5 or utils[least_utilized].gpu == 0:
        #     with self._lock:
        #         self._counter = (self._counter + 1) % len(self.lms)
        #     return self._counter
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
                "num_ctx": CONTEXT_WINDOW}
MODELS = []
for i in range(MAX_RETRIES + 1):
    temperature = get_temperature(i)
    MODEL_KWARGS["temperature"] = temperature
    MODEL_KWARGS["seed"] = i
    MODELS.append(ThreadSafeOllamaLM(**MODEL_KWARGS))

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
    month: int | None
    day_of_month: int | None
    week: int | None


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
Include ALL mentions from the following list (and any additional mentions found in the text):\n""" + "\n".join(
        sorted(CHEMO_DRUGS))
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


class ChemoTimelineBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.debrief_lm = dspy.ChainOfThought(ChemoNotesTimeline)
        self.update_lm = dspy.ChainOfThought(ChemoTimelineUpdate)

    def retry(self, func, **kwargs):
        for model in MODELS:
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

        return sorted(unique_events,
                      key=lambda x: (x[2].year, x[2].month or 0, x[2].day_of_month or 0,
                                     x[2].week or 0, x[0], x[1]))  # Sort by date, then by drug and relation

    def evaluate_and_update_timeline(self, timeline, content, reasoning):
        """Process content and update timeline"""
        # Extract chemotherapy events from content
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

        return timeline

    def forward(self, chunks: list[str]):
        """Build timeline from text chunks"""
        # Initialize
        timeline = []
        reasoning = []

        # Process chunks
        for chunk in chunks:
            timeline = self.evaluate_and_update_timeline(timeline, chunk, reasoning)

        # Convert dates to strings
        timeline = [(drug, relation, convert_date_to_string(date)) for drug, relation, date in timeline]

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


def add_to_content(ent_or_rel, note, content, needs_id):
    if isinstance(ent_or_rel, dict):
        del ent_or_rel['parentsType']  # always "TemporalRelations"
        if ent_or_rel['properties'] is None:
            del ent_or_rel['properties']
        else:
            for key, value in list(ent_or_rel['properties'].items()):
                if value == "N/A":
                    del ent_or_rel['properties'][key]
        if needs_id:
            ent_or_rel['id'] = int(ent_or_rel['id'].split("@")[0])  # Remove @<id> suffix
        else:
            del ent_or_rel['id']
        if "span" in ent_or_rel:
            assert "text" not in ent_or_rel, "Entity already has text field"
            start, end = [int(x) for x in ent_or_rel['span'].split(',')]
            # del ent_or_rel['span']
            ent_or_rel['text'] = note[start:end]
        content += json.dumps(ent_or_rel, indent=2) + "\n\n"
    return content


if __name__ == "__main__":
    import json
    import math
    import os
    from collections import defaultdict
    from copy import deepcopy
    from functools import partial
    from tqdm import tqdm

    task1_path = "../../chemoTimelines2024_train_dev_labeled/subtask1"

    notes_path = os.path.join(task1_path, "Patient_Notes")
    xml_path = os.path.join(task1_path, "Gold_PairWise_Annotations")
    timelines_path = os.path.join(task1_path, "Gold_Timelines_allPatients_processed")

    data = {split: {"chunks": defaultdict(list), "timeline": {}} for split in ["train", "dev"]}

    # Load timelines and notes
    report_type = lambda x: x.split('_')[-1].strip(".txt").upper()
    for file in os.listdir(timelines_path):
        if file.endswith(".json"):
            split, site, patient = file[:-5].split("_")
            with open(os.path.join(timelines_path, file), 'r') as f:
                timelines = json.load(f)
            data[split]["timeline"][f"{site}_{patient}"] = [tuple(item) for item in timelines]
            reports_path = os.path.join(notes_path, site, split, patient)
            for report in sorted(os.listdir(reports_path),
                                 key=lambda x: ['NOTE', 'PGN', 'RAD', report_type(x)].index(report_type(x))):
                with open(os.path.join(reports_path, report), 'r') as f:
                    note = f.read().strip()
                report_id = report[:-4]
                xml_file = os.path.join(xml_path, site, split, f"{site}_{patient}_{split}", report_id,
                                        f"{report_id}.Temporal_Relation.gold.inprogress.xml")
                content = ""
                try:
                    with open(xml_file, 'r') as f:
                        xml_content = f.read()
                    xml_dict = xmltodict.parse(xml_content)['data']['annotations']
                    has_relation = 'relation' in xml_dict
                    for ent in xml_dict['entity']:
                        content = add_to_content(ent, note, content, has_relation)
                    for rel in xml_dict.get('relation', []):
                        content = add_to_content(rel, note, content, False)
                except FileNotFoundError:
                    pass
                if content:
                    content = "Temporal relations:\n\n" + content
                    content += "\n\n\n"
                content += note
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
