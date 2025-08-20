import json
import math
import os
import random
import re
from functools import lru_cache

import dspy
import tiktoken
import xmltodict
from typing_extensions import NamedTuple, Literal, Any

from eval_for_optim import evaluation_f1
from mychatadapter import *
from threadsafe_ollama import create_threadsafe_models
from config import Config, MODEL, CONTEXT_WINDOW, MIN_TEMPERATURE, MAX_TEMPERATURE, MAX_RETRIES

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

# Create thread-safe models with different temperatures
MODELS = create_threadsafe_models(
    model_name=MODEL,
    context_window=CONTEXT_WINDOW,
    temperature_range=(MIN_TEMPERATURE, MAX_TEMPERATURE),
    num_models=MAX_RETRIES + 1
)

dspy.configure(lm=MODELS[0], adapter=MyChatAdapter())


class PatientReport(NamedTuple):
    ID: int
    type: str
    text: str
    temporal_relations: list[list[tuple[str, Any]]]


def convert_date_to_string(date: Date) -> str:
    """Convert Date tuple to string in competition format."""
    assert date.year is not None
    if date.week_of_year is None or date.day_of_month is not None:
        if date.month is None:
            return f"{date.year}"
        if date.day_of_month is None:
            return f"{date.year}-{date.month:02d}"
        return f"{date.year}-{date.month:02d}-{date.day_of_month:02d}"
    return f"{date.year}-w{date.week_of_year:02d}"


class SACTTimelineUpdate(dspy.Signature):
    """
    Update SACT timeline based on patient reports.
    (docstring unchanged…)
    """
    reports: list[PatientReport] = dspy.InputField(desc="JSON with clinical text and temporal relations")
    timeline: Timeline = dspy.InputField(desc="existing events in the timeline")
    update: Update = dspy.OutputField(desc="update to the timeline, with 'add' and 'remove' lists of events")


class SACTTimelineBuilder(dspy.Module):
    def __init__(self, max_iter=3, max_reports=10):
        super().__init__()
        self.update_lm = dspy.ChainOfThought(SACTTimelineUpdate)
        self.max_iter = max_iter
        self.max_reports = max_reports

    def retry(self, func, **kwargs):
        for model in MODELS:
            with dspy.context(lm=model, adapter=MyChatAdapter()):
                output = func(**kwargs)

            if not any(x is None for x in output.values()):
                return output

        if output["update"] is None:
            output["update"] = Update(add=[], remove=[])
            output["reasoning"] = "No update generated, returning empty update."
        return output

    def process_timeline_update(self, current_timeline, update):
        """Merge new events with current timeline"""
        current_timeline = [event for event in current_timeline if event not in update.remove]

        if not update.add:
            return current_timeline

        all_events = current_timeline + update.add

        seen = set()
        unique_events = []
        for event in all_events:
            if event not in seen:
                seen.add(event)
                unique_events.append(event)

        return sorted(unique_events,
                      key=lambda x: (x[2].year, x[2].month or 0, x[2].day_of_month or 0,
                                     x[2].week_of_year or 0, x[0], x[1]))

    def evaluate_and_update_timeline(self, timeline, content, reasoning):
        output = self.retry(self.update_lm, timeline=timeline, reports=content)
        reasoning.append(output.get("reasoning", ""))
        timeline = self.process_timeline_update(timeline, output["update"])
        return timeline

    def forward(self, reports: list[PatientReport]) -> dspy.Prediction:
        timeline = []
        reasoning = []
        random.seed(42)

        report_clumps = self._create_report_clumps(reports, target_size=CONTEXT_WINDOW // 8)
        max_iter = math.ceil(self.max_iter * min(self.max_reports / len(report_clumps), 1))

        for i in range(max_iter):
            random.shuffle(report_clumps)
            for clump in tqdm(report_clumps, desc=f"Processing report clumps (iteration {i + 1}/{max_iter})"):
                timeline = self.evaluate_and_update_timeline(timeline, clump, reasoning)
            if not timeline:
                break

        timeline = [(drug, relation, convert_date_to_string(date)) for drug, relation, date in timeline]

        return dspy.Prediction(
            timeline=timeline,
            reasoning="\n\n\n".join(reasoning)
        )

    def _create_report_clumps(self, reports: list[PatientReport], target_size: int) -> list[list[PatientReport]]:
        total_tokens = get_token_count(reports)
        if len(reports) == 1 or total_tokens < target_size:
            return [reports]

        report_groups = {}
        for report in reports:
            report_groups.setdefault(report.ID, []).append(report)

        sorted_groups = [report_groups[id] for id in sorted(report_groups.keys())]

        clumps = []
        current_clump = sorted_groups[0] if sorted_groups else []
        target_per_clump = math.ceil(total_tokens / math.ceil(total_tokens / target_size))

        for group in sorted_groups[1:]:
            test_clump = current_clump + group
            if get_token_count(test_clump) < target_per_clump:
                current_clump = test_clump
            else:
                clumps.append(current_clump)
                current_clump = group

        if current_clump:
            clumps.append(current_clump)

        return clumps


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
        if not pred.timeline and not example.timeline:
            print("Both predicted and true timelines are empty.")
            return 1.0

        f1_score = evaluation_f1(example.timeline, pred.timeline, strict=True)

        pred_set = set(pred.timeline)
        true_set = set(example.timeline)

        print("Overlap:", pred_set & true_set)
        print("Missing:", true_set - pred_set)
        print("Extraneous:", pred_set - true_set)
        print(f"F1 Score: {f1_score}")

        return f1_score

    evaluator = dspy.Evaluate(devset=dev,
                              metric=timeline_f1,
                              num_threads=1,
                              display_progress=True,
                              provide_traceback=True,
                              max_errors=0)

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
        print("   export CHEMO_ENABLE_PROMPT_OPTIMIZATION=true")
        print("   export CHEMO_PROMPT_OPTIMIZER=simba")
        print("Skipping optimization.")
        return acc_zero, outputs_zero, zeroshot

    optimizer = get_prompt_optimizer(Config.PROMPT_OPTIMIZER, metric=timeline_f1)
    fewshot = optimizer.compile(zeroshot, trainset=train)

    print("Few-shot results:")
    evaluation = evaluator(fewshot)
    acc_few, outputs_few = evaluation["score"], evaluation["results"]
    print(f"Few-shot accuracy: {acc_few}")

    if acc_zero < acc_few:
        fewshot.save("fewshot_model.json")
        print("Saved improved few-shot model")
        return acc_few, outputs_few, fewshot
    else:
        print("Few-shot model did not improve over zero-shot model")
        return acc_zero, outputs_zero, zeroshot


def add_to_temporal_relations(ent_rel, text, temporal_relations, needs_id):
    if isinstance(ent_rel, dict):
        del ent_rel['parentsType']
        if ent_rel['properties'] is None:
            del ent_rel['properties']
        else:
            for key, value in list(ent_rel['properties'].items()):
                if value == "N/A":
                    del ent_rel['properties'][key]
        ent_rel['id'] = ent_rel['id'].replace("@gold", "")
        if "span" in ent_rel:
            assert "text" not in ent_rel
            start, end = [int(x) for x in ent_rel['span'].split(',')]
            ent_rel['text'] = text[start:end]
        temporal_relations.append(sorted(ent_rel.items()))


if __name__ == "__main__":
    import argparse
    import json
    from collections import defaultdict
    from copy import deepcopy
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Run Task1 timeline generation')
    parser.add_argument('--output-dir', default='.', help='Output directory for generated timelines (default: current directory)')
    args = parser.parse_args()

    task1_path = "/data/project/alstate/chemoTimeline2025/chemoTimelines2024_train_dev_labeled/subtask1"

    notes_path = os.path.join(task1_path, "Patient_Notes")
    xml_path = os.path.join(task1_path, "Gold_PairWise_Annotations")
    timelines_path = os.path.join(task1_path, "Gold_Timelines_allPatients_processed")

    data = {split: {"reports": defaultdict(list), "timeline": {}} for split in ["train", "dev"]}

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
                    text = f.read().strip()
                report_name = report[:-4]
                ID = int(re.search(r'report(\d+)', report).group(1))
                xml_file = os.path.join(xml_path, site, split, f"{site}_{patient}_{split}", report_name,
                                        f"{report_name}.Temporal_Relation.gold.inprogress.xml")
                temporal_relations = []
                try:
                    with open(xml_file, 'r') as f:
                        xml_content = f.read()
                    xml_dict = xmltodict.parse(xml_content)['data']['annotations']
                    has_relation = 'relation' in xml_dict
                    for ent in xml_dict['entity']:
                        add_to_temporal_relations(ent, text, temporal_relations, has_relation)
                    for rel in xml_dict.get('relation', []):
                        add_to_temporal_relations(rel, text, temporal_relations, False)
                    print(f"Loaded {len(temporal_relations)} temporal relations from {xml_file}")
                except FileNotFoundError:
                    pass
                data[split]["reports"][f"{site}_{patient}"].append(
                    PatientReport(ID=ID, type=report_type(report), text=text,
                                  temporal_relations=temporal_relations))

    for split in data:
        new_data = {}
        for patient, reports in data[split]["reports"].items():
            new_data[patient] = dspy.Example({
                "reports": reports,
                "timeline": data[split]["timeline"][patient]
            }).with_inputs("reports")
        if split == "train":
            subset = {}
            # Pre-compute sites to avoid repeated string operations
            sites = set(key.split('_')[0] for key in new_data.keys())
            for site in sites:
                site_patients = [key for key in new_data.keys() if key.startswith(site)]
                # Cache token counts and sort by pre-computed values
                patient_tokens = [(key, get_token_count(new_data[key]["reports"]) if new_data[key]["timeline"] else float('inf')) for key in site_patients]
                patient_tokens.sort(key=lambda x: x[1])
                subset.update({key: new_data[key] for key, _ in patient_tokens[:20]})
            new_data = subset
        data[split] = new_data

    train_data = list(data["train"].values())
    dev_data = list(data["dev"].values())

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")

    # Warn user about optimization status
    Config.warn_if_optimization_disabled()

    _, _, builder = evaluate(train_data, dev_data, SACTTimelineBuilder())

    jsons = defaultdict(dict)
    for split in ["train", "dev"]:
        for key, value in tqdm(data[split].items(), desc=f"Processing {split} data"):
            site, patient = key.split('_')
            generated = builder(value["reports"])
            offset = 0
            for i, entry in enumerate(list(generated.timeline)):
                if not re.match(r'^\d{4}(?:-(?:\d{2}(?:-\d{2})?|w\d{2}))?$', entry[2]):
                    print(f"Invalid date format in entry {entry} for {patient} in {site} {split}. Removing entry.")
                    generated.timeline.remove(entry)
                    offset += 1
                else:
                    generated.timeline[i - offset] = (entry[0], entry[1], entry[2].replace("-00", ""))
            jsons[f"{site}_{split}"][patient] = generated.timeline
    
    # Create model postfix by cleaning up the model name
    model_postfix = MODEL.replace('/', '_').replace(':', '_')
    task_postfix = "task1"
    
    for site_split, timelines in jsons.items():
        filename = f"{site_split}_all_patients_generated_timelines_{model_postfix}_{task_postfix}.json"
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(timelines, f, indent=2)

    print("Timeline examples created successfully.")

