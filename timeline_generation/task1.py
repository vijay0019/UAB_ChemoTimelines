import math
import os
import random
import re

import dspy
import tiktoken
import xmltodict
from typing_extensions import NamedTuple, Literal, Any

from eval_for_optim import evaluation_f1
from mychatadapter import *
from threadsafe_ollama import create_threadsafe_models
from config import Config, MODEL, CONTEXT_WINDOW, MIN_TEMPERATURE, MAX_TEMPERATURE, MAX_RETRIES

tokenizer = tiktoken.get_encoding("cl100k_base")

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
        total_tokens = len(tokenizer.encode(str(reports)))
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
            if len(tokenizer.encode(str(test_clump))) < target_per_clump:
                current_clump = test_clump
            else:
                clumps.append(current_clump)
                current_clump = group

        if current_clump:
            clumps.append(current_clump)

        return clumps


def evaluate(train, dev, zeroshot, optimize=True):
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
                              return_outputs=True,
                              provide_traceback=True,
                              max_errors=0)

    print("Zero-shot results:")
    evaluation = evaluator(zeroshot)
    acc_zero, outputs_zero = evaluation["score"], evaluation["results"]
    print(f"Zero-shot accuracy: {acc_zero}")

    if not optimize:
        print("Skipping optimization.")
        return acc_zero, outputs_zero, zeroshot

    optimizer = dspy.SIMBA(metric=timeline_f1, bsize=10, num_candidates=4, max_steps=4, num_threads=1)
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
    import json
    from collections import defaultdict
    from copy import deepcopy
    from tqdm import tqdm

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
            for site in set(key.split('_')[0] for key in new_data.keys()):
                site_patients = [key for key in new_data.keys() if key.startswith(site)]
                site_patients.sort(key=lambda x: len(tokenizer.encode(str(new_data[x]["reports"]))) if new_data[x]["timeline"] else float('inf'))
                subset.update({key: new_data[key] for key in site_patients[:20]})
            new_data = subset
        data[split] = new_data

    train_data = list(data["train"].values())
    dev_data = list(data["dev"].values())

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")

    _, _, builder = evaluate(train_data, dev_data, SACTTimelineBuilder(), optimize=True)

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
    for site_split, timelines in jsons.items():
        with open(f"{site_split}_all_patients_generated_timelines.json", "w") as f:
            json.dump(timelines, f, indent=2)

    print("Timeline examples created successfully.")

