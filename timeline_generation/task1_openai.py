import os
import random
import re
from time import sleep

import dspy
import pandas as pd
import tiktoken
import xmltodict
from typing_extensions import NamedTuple, Literal

from mychatadapter import MyChatAdapter

tokenizer = tiktoken.get_encoding("cl100k_base")

MODEL = "azure/gpt-4.1-mini"
CONTEXT_WINDOW = 1048576
MAX_TOKENS = 32768
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 0.2
MAX_RETRIES = 2


def get_temperature(retry_count):
    return MIN_TEMPERATURE + (
        (MAX_TEMPERATURE - MIN_TEMPERATURE) * retry_count / MAX_RETRIES if MAX_RETRIES > 0 else 0.0)


MODEL_KWARGS = {
    "api_base": os.environ["AZURE_API_BASE"],
	"api_version": os.environ["AZURE_API_VERSION"],
	"api_key": os.environ["AZURE_API_KEY"],
	"model": MODEL,
	"max_tokens": MAX_TOKENS
}
MODELS = []
for i in range(MAX_RETRIES + 1):
    temperature = get_temperature(i)
    MODEL_KWARGS["temperature"] = temperature
    MODEL_KWARGS["seed"] = i
    MODELS.append(dspy.LM(**MODEL_KWARGS))

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


class PatientReport(NamedTuple):
    ID: int
    type: str
    text: str
    temporal_relations: list[dict]


class ChemoTimelineUpdate(dspy.Signature):
    """
Update chemotherapy timeline based on patient reports.
Exclude surgical procedures, radiation therapy, and other non-chemotherapy-related events.

Drug Names: Extract drug names EXACTLY as they appear in the clinical text. Do NOT normalize or convert to generic names. 
Include ALL variations found in the text:
- Brand names (cytoxan, taxotere, abraxane)
- Generic names (cyclophosphamide, docetaxel, paclitaxel) 
- Abbreviations (tc, ac, a/c)
- Generic terms (chemotherapy, chemo)
- Slight variations/typos (docetaxol for docetaxel)

If the text mentions both "cyclophosphamide" and "cytoxan", include both as separate entries.
If the text mentions both "chemotherapy" and specific drug names, include both.

Relations:
- 'begins-on': treatment/medication starts
- 'ends-on': treatment/medication ends  
- 'contains-1': treatment occurred within timeframe

Acceptable date formats:
1. Specify year, month, and day.
2. Specify year and week.
3. Specify year and month.
4. Specify year only.

Example output format:
[[ ## add ## ]]
[
    ('tc', 'contains-1', Date(year=2011, month=8, day_of_month=None, week=None)),
    ('cyclophosphamide', 'begins-on', Date(year=2011, month=8, day_of_month=1, week=None)),
    ('cytoxan', 'begins-on', Date(year=2011, month=8, day_of_month=1, week=None)),
    ('cyclophosphamide', 'contains-1', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('cytoxan', 'contains-1', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('docetaxel', 'begins-on', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('docetaxol', 'begins-on', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('taxotere', 'begins-on', Date(year=2011, month=8, day_of_month=8, week=None)),
    ('chemo', 'contains-1', Date(year=2011, month=8, day_of_month=10, week=None)),
    ('chemotherapy', 'contains-1', Date(year=2011, month=8, day_of_month=10, week=None)),
    ('chemotherapy', 'ends-on', Date(year=2012, month=None, day_of_month=None, week=None))
]
[[ ## remove ## ]]
[
    ('cyclophosphamide', 'begins-on', Date(year=2011, month=8, day_of_month=8, week=None))
]
[[ ## completed ## ]]
	"""
    reports: list[PatientReport] = dspy.InputField(desc="JSON with clinical text and temporal relations")
    timeline: TIMELINE = dspy.InputField(desc="existing events in the timeline")
    add: TIMELINE = dspy.OutputField(desc="chemotherapy events to add")
    remove: TIMELINE = dspy.OutputField(desc="chemotherapy events to remove")


class ChemoTimelineBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.update_lm = dspy.ChainOfThought(ChemoTimelineUpdate)

    def retry(self, func, **kwargs):
        for model in MODELS:
            with dspy.context(lm=model, adapter=MyChatAdapter()):
                output = func(**kwargs)

            if not any(x is None for x in output.values()):
                return output

        raise RuntimeError(f"Failed to get valid response after {MAX_RETRIES} retries.")

    def process_timeline_update(self, current_timeline, add, remove):
        """Merge new events with current timeline"""
        # Remove specified events
        current_timeline = [event for event in current_timeline if event not in remove]

        if not add:
            return current_timeline

        # Combine with existing timeline
        all_events = current_timeline + add

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
        # Update timeline with new content
        output = self.retry(self.update_lm, timeline=timeline, reports=content)
        if output.get("reasoning"):
            reasoning.append(output["reasoning"])

        # Process the update
        timeline = self.process_timeline_update(timeline, output["add"], output["remove"])

        return timeline

    def forward(self, reports: list[str]):
        """Build timeline from text reports"""
        # Initialize
        timeline = []
        reasoning = []

        # Process reports
        for report in reports:
            timeline = self.evaluate_and_update_timeline(timeline, report, reasoning)

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
    # val = [example for example in train if sum([len(tokenizer.encode(report)) for report in example.reports]) >= CONTEXT_WINDOW or len(example.timeline) == 0]
    # train = [example for example in train if sum([len(tokenizer.encode(report)) for report in example.reports]) < CONTEXT_WINDOW and len(example.timeline) > 0]
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


def add_to_temporal_relations(ent_rel, text, temporal_relations, needs_id):
    if isinstance(ent_rel, dict):
        del ent_rel['parentsType']  # always "TemporalRelations"
        if ent_rel['properties'] is None:
            del ent_rel['properties']
        else:
            for key, value in list(ent_rel['properties'].items()):
                if value == "N/A":
                    del ent_rel['properties'][key]
        if needs_id:
            ent_rel['id'] = int(ent_rel['id'].split("@")[0])  # Remove @<id> suffix
        else:
            del ent_rel['id']
        if "span" in ent_rel:
            assert "text" not in ent_rel, "Already has text field"
            start, end = [int(x) for x in ent_rel['span'].split(',')]
            # del ent_rel['span']
            ent_rel['text'] = text[start:end]
        temporal_relations.append(ent_rel)


def concatenate_reports(data, target):
    data = data.copy()
    # Concatenate reports intelligently to fit within the context window
    for split in data:
        items = list(data[split]["reports"].items())
        for key, value in items:
            # Concatenate reports if they fit within the context window
            n = len(tokenizer.encode(str(value)))
            if len(value) == 1 or n < target:
                data[split]["reports"][key] = [value]
            # If more than one report, concatenate them intelligently
            else:
                # Group reports by report ID
                report_groups = [[v for v in value if v.ID == i] for i in sorted(set(v.ID for v in value))]
                # Concatenate groups while respecting the context window
                # Use a greedy approach to concatenate as many groups as possible without exceeding the context window
                # This is a simple heuristic and may not be optimal
                concatenated_groups = []
                current_group = report_groups[0]
                # target = math.ceil(n / math.ceil(2 * n / CONTEXT_WINDOW))
                for report_group in report_groups[1:]:
                    # Check if adding the next group exceeds the context window
                    if len(tokenizer.encode(str(current_group + report_group))) < target:
                        current_group += report_group
                    else:
                        concatenated_groups.append(current_group)
                        current_group = report_group
                concatenated_groups.append(current_group)
                data[split]["reports"][key] = concatenated_groups

    # Rearrange data structure
    for split in data:
        new_data = {}
        for patient, reports in data[split]["reports"].items():
            if patient in data[split]["timeline"]:
                new_data[patient] = dspy.Example({
                    "reports": reports,
                    "timeline": data[split]["timeline"][patient]
                }).with_inputs("reports")
        data[split] = new_data

    return data


if __name__ == "__main__":
    import json
    from collections import defaultdict
    from copy import deepcopy
    from tqdm import tqdm

    task1_path = "../../chemoTimelines2024_train_dev_labeled/subtask1"

    notes_path = os.path.join(task1_path, "Patient_Notes")
    xml_path = os.path.join(task1_path, "Gold_PairWise_Annotations")
    timelines_path = os.path.join(task1_path, "Gold_Timelines_allPatients_processed")

    data = {split: {"reports": defaultdict(list), "timeline": {}} for split in ["train", "dev"]}

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
                except FileNotFoundError:
                    pass
                data[split]["reports"][f"{site}_{patient}"].append(
                    PatientReport(ID=ID, type=report_type(report), text=text,
                                  temporal_relations=temporal_relations))

    data_old = deepcopy(data)
    data = concatenate_reports(data, target=CONTEXT_WINDOW * 0.0625)  # 1/16 of the context window

    train_data = list(data["train"].values())
    dev_data = list(data["dev"].values())

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")

    # acc_large, outputs_large, zeroshot = evaluate(train_data, dev_data, ChemoTimelineBuilder(), optimize=False)
    # data = concatenate_reports(data_old, target=CONTEXT_WINDOW * 0.25)
    # train_data = list(data["train"].values())
    # dev_data = list(data["dev"].values())
    # acc_small, outputs_small, fewshot = evaluate(train_data, dev_data, ChemoTimelineBuilder())
    # if acc_small > acc_large:
    #     print(f"Small context window model ({acc_small}) outperformed large context window model ({acc_large}).")
    #     builder = zeroshot
    # else:
    #     print(f"Large context window model ({acc_large}) outperformed small context window model ({acc_small}).")
    #     builder = fewshot

    # _, _, builder = evaluate(train_data, dev_data, ChemoTimelineBuilder(), optimize=True)
    builder = ChemoTimelineBuilder()

    jsons = defaultdict(dict)
    for split in ["train", "dev"]:
        if split != "dev":
            continue
        for key, value in tqdm(data[split].items(), desc=f"Processing {split} data"):
            site, patient = key.split('_')
            if site != "melanoma":
                continue
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
