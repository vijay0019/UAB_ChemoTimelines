import math
import os
import random
import re
from time import sleep

import dspy
import pynvml
import tiktoken
import xmltodict
from typing_extensions import NamedTuple, Literal, Any

from eval_for_optim import evaluation_f1
from mychatadapter import *

tokenizer = tiktoken.get_encoding("cl100k_base")

MODEL = "ollama/phi4:latest"
CONTEXT_WINDOW = 16384
MIN_TEMPERATURE = 0.2
MAX_TEMPERATURE = 0.4
MAX_RETRIES = 2


class ThreadSafeOllamaLM(dspy.LM):
    def __init__(self, ports=(11438,), **kwargs):
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
        sleep(random.random() * 0.1)  # small delay to avoid contention
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


pynvml.nvmlInit()
deviceCount = 1#pynvml.nvmlDeviceGetCount()


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

SACT is defined as follows:
"Systemic anticancer therapy (SACT), which includes traditional cytotoxic chemotherapy, endocrine therapy, targeted therapy, and immunotherapy, has both a low therapeutic index as well as synergistic potential when agents are given in combination."

Drug Names: Extract drug names EXACTLY as they appear in the clinical text (except be sure to put them in lowercase). Do NOT normalize or convert to generic names. 
Include ALL variations found in the text:
- Brand names (cytoxan, taxotere, abraxane)
- Generic names (cyclophosphamide, docetaxel, paclitaxel) 
- Abbreviations (tc, ac, a/c)
- Generic terms (chemotherapy, chemo)
- Slight variations/typos (docetaxol for docetaxel)

If the text mentions both "cyclophosphamide" and "cytoxan", include both as separate entries.
If the text mentions both "chemotherapy" and specific drug names, include both.
Only include drugs that have a temporal relation in the text.

Relations:
- 'begins-on': treatment/medication starts
- 'ends-on': treatment/medication ends  
- 'contains-1': treatment occurred within timeframe
'begins-on' and 'ends-on' supersede 'contains-1' for the same drug/date combination. Only use them if the text explicitly states the start or end date of the treatment.

Acceptable date formats:
1. Specify year, month, and day.
2. Specify year and week.
3. Specify year and month.
4. Specify year only.
Try to be as specific as possible, but do not invent dates that are not mentioned in the text.

Keep in mind that the reports are only a subset of the full timeline, so there may be events in the timeline that are not mentioned in the reports. Do not remove events simply because they are not mentioned in the reports.

If a report doesn't have temporal relations, that likely means the report does not contain any relevant information for the timeline. Avoid adding events based solely on hypothetical or planned mentions without temporal grounding.

Example output format:
[[ ## update ## ]]
Update(
    add=[
        ('tc', 'contains-1', Date(year=2011, month=8, day_of_month=None, week_of_year=None)),
        ('cyclophosphamide', 'begins-on', Date(year=2011, month=8, day_of_month=1, week_of_year=None)),
        ('cytoxan', 'contains-1', Date(year=2011, month=8, day_of_month=1, week_of_year=None)),
        ('cyclophosphamide', 'contains-1', Date(year=2011, month=8, day_of_month=8, week_of_year=None)),
        ('cytoxan', 'contains-1', Date(year=2011, month=8, day_of_month=8, week_of_year=None)),
        ('docetaxel', 'contains-1', Date(year=2011, month=8, day_of_month=8, week_of_year=None)),
        ('docetaxol', 'contains-1', Date(year=2011, month=8, day_of_month=8, week_of_year=None)),
        ('taxotere', 'contains-1', Date(year=2011, month=8, day_of_month=8, week_of_year=None)),
        ('chemo', 'contains-1', Date(year=2011, month=8, day_of_month=10, week_of_year=None)),
        ('chemotherapy', 'contains-1', Date(year=2011, month=8, day_of_month=10, week_of_year=None)),
        ('taxotere', 'contains-1', Date(year=2011, month=None, day_of_month=None, week_of_year=36)),
        ('chemotherapy', 'ends-on', Date(year=2012, month=None, day_of_month=None, week_of_year=None))
    ],
    remove=[
        ('cyclophosphamide', 'begins-on', Date(year=2011, month=8, day_of_month=8, week_of_year=None))
    ]
)
[[ ## completed ## ]]
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

        # raise RuntimeError(f"Failed to get valid response after {MAX_RETRIES} retries.")
        if output["update"] is None:
            output["update"] = Update(add=[], remove=[])
        return output

    def process_timeline_update(self, current_timeline, update):
        """Merge new events with current timeline"""
        # Remove specified events
        current_timeline = [event for event in current_timeline if event not in update.remove]

        if not update.add:
            return current_timeline

        # Combine with existing timeline
        all_events = current_timeline + update.add

        # Remove duplicates while preserving order
        seen = set()
        unique_events = []
        for event in all_events:
            if event not in seen:
                seen.add(event)
                unique_events.append(event)

        return sorted(unique_events,
                      key=lambda x: (x[2].year, x[2].month or 0, x[2].day_of_month or 0,
                                     x[2].week_of_year or 0, x[0], x[1]))  # Sort by date, then by drug and relation

    def evaluate_and_update_timeline(self, timeline, content, reasoning):
        """Process content and update timeline"""
        # Update timeline with new content
        output = self.retry(self.update_lm, timeline=timeline, reports=content)
        reasoning.append(output.get("reasoning", ""))

        # Process the update
        timeline = self.process_timeline_update(timeline, output["update"])

        return timeline

    def forward(self, reports: list[PatientReport]) -> dspy.Prediction:
        """Build timeline from text reports"""
        # Initialize
        timeline = []
        reasoning = []
        
        # NEW: Track all timelines from each iteration
        all_timelines = []
        
        random.seed(42)
        
        # Create clumps of reports that fit within context window
        report_clumps = self._create_report_clumps(reports, target_size=CONTEXT_WINDOW // 8)
        
        # Limit iterations based on max_reports
        max_iter = math.ceil(self.max_iter * min(self.max_reports / len(report_clumps), 1))

        # Process report clumps
        for i in range(max_iter):
            random.shuffle(report_clumps)  # Shuffle clumps to avoid bias
            iteration_timeline = []
            for clump in tqdm(report_clumps, desc=f"Processing report clumps (iteration {i + 1}/{max_iter})"):
                iteration_timeline = self.evaluate_and_update_timeline(iteration_timeline, clump, reasoning)
            
            if not iteration_timeline:
                break # If no timeline was generated, stop early
                
            # NEW: Store this iteration's timeline
            all_timelines.append(iteration_timeline.copy())

        # NEW: Filter events by frequency across iterations
        timeline = self.filter_frequent_events(all_timelines)

        # Convert dates to strings
        timeline = [(drug, relation, convert_date_to_string(date)) for drug, relation, date in timeline]

        return dspy.Prediction(
            timeline=timeline,
            reasoning="\n\n\n".join(reasoning)
        )

    def filter_frequent_events(self, all_timelines, min_frequency=2):
        """Keep only events that appear in multiple iterations"""
        from collections import Counter
        
        if len(all_timelines) <= 1:
            return all_timelines[0] if all_timelines else []
        
        # Count how many times each event appears
        event_counts = Counter()
        for timeline in all_timelines:
            for event in timeline:
                event_counts[event] += 1
        
        # Keep events that appear at least min_frequency times
        # OR if we have very few iterations, keep events that appear in majority
        threshold = min(min_frequency, max(1, len(all_timelines) // 2))
        
        frequent_events = [event for event, count in event_counts.items() 
                        if count >= threshold]
        
        # Sort by date, then by drug and relation
        frequent_events.sort(
            key=lambda x: (x[2].year, x[2].month or 0, x[2].day_of_month or 0,
                        x[2].week_of_year or 0, x[0], x[1])
        )
        
        return frequent_events
    
    def _create_report_clumps(self, reports: list[PatientReport], target_size: int) -> list[list[PatientReport]]:
        """Create clumps of reports that fit within the target token size"""
        # Check if all reports fit in one clump
        total_tokens = len(tokenizer.encode(str(reports)))
        if len(reports) == 1 or total_tokens < target_size:
            return [reports]
        
        # Group reports by report ID
        report_groups = {}
        for report in reports:
            if report.ID not in report_groups:
                report_groups[report.ID] = []
            report_groups[report.ID].append(report)
        
        # Sort groups by ID for consistent processing
        sorted_groups = [report_groups[id] for id in sorted(report_groups.keys())]
        
        # Use greedy approach to create clumps within target size
        clumps = []
        current_clump = sorted_groups[0] if sorted_groups else []
        target_per_clump = math.ceil(total_tokens / math.ceil(total_tokens / target_size))
        
        for group in sorted_groups[1:]:
            # Check if adding the next group exceeds the target size
            test_clump = current_clump + group
            if len(tokenizer.encode(str(test_clump))) < target_per_clump:
                current_clump = test_clump
            else:
                clumps.append(current_clump)
                current_clump = group
        
        # Add the last clump
        if current_clump:
            clumps.append(current_clump)
        
        return clumps


def evaluate(train, dev, zeroshot, optimize=True):
    def timeline_f1(example, pred, trace=None):
        """Enhanced timeline evaluation using the official evaluation logic"""
        if not pred.timeline and not example.timeline:
            print("Both predicted and true timelines are empty.")
            return 1.0

        # Use the official evaluation function with strict evaluation
        f1_score = evaluation_f1(example.timeline, pred.timeline, strict=True)
        
        # Convert to sets for debugging output
        pred_set = set(pred.timeline)
        true_set = set(example.timeline)
        
        print("Overlap:", pred_set & true_set)
        print("Missing:", true_set - pred_set)
        print("Extraneous:", pred_set - true_set)
        print(f"F1 Score: {f1_score}")

        return f1_score

    # # Split train into train and validation sets
    # val = [example for example in train if sum([len(tokenizer.encode(report)) for report in example.reports]) >= CONTEXT_WINDOW or len(example.timeline) == 0]
    # train = [example for example in train if sum([len(tokenizer.encode(report)) for report in example.reports]) < CONTEXT_WINDOW and len(example.timeline) > 0]
    # print(f"Train examples: {len(train)}, Validation examples: {len(val)}")

    # Define evaluator
    evaluator = dspy.Evaluate(devset=dev,
                              metric=timeline_f1,
                              num_threads=1,
                              display_progress=True,
                              return_outputs=True,
                              provide_traceback=True,
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
    optimizer = dspy.SIMBA(metric=timeline_f1, bsize=10, num_candidates=4, max_steps=4, num_threads=1)
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
        # if needs_id:
        #     ent_rel['id'] = int(ent_rel['id'].split("@")[0])  # Remove @<id> suffix
        # else:
        #     del ent_rel['id']
        ent_rel['id'] = ent_rel['id'].replace("@gold", "")  # Remove @gold suffix
        if "span" in ent_rel:
            assert "text" not in ent_rel, "Already has text field"
            start, end = [int(x) for x in ent_rel['span'].split(',')]
            # del ent_rel['span']
            ent_rel['text'] = text[start:end]
        temporal_relations.append(sorted(ent_rel.items()))  # Sort items to ensure consistent order


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
                    print(f"Loaded {len(temporal_relations)} temporal relations from {xml_file}")
                except FileNotFoundError:
                    pass
                data[split]["reports"][f"{site}_{patient}"].append(
                    PatientReport(ID=ID, type=report_type(report), text=text,
                                  temporal_relations=temporal_relations))

    # Rearrange data structure
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
                # select top 20 patients with shortest total report length
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
