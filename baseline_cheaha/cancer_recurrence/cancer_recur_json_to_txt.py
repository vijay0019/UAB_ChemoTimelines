"""
Convert the cancer recurrence JSON

the output file format is:
    batch{#}/{mrn}/{mrn}_{reportid}_{sources}.txt

"""

# may swap to json-lineage, if performance is a problem

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Final

# 3.5 GB file, so a CPU with 16 GB would probably be sufficient
CANCER_RECUR_JSON_FILE: Final[Path] = Path("/data/user/ozborn/CancerRecurrenceUAB/to_deep_south/nrm_rcs.json")
DESTINATION_DIR: Final[Path] = Path("~/prj/chemotimelines/baseline_cheaha/cancer_recur_batches/").expanduser()


def write_batch_to_disk(batch: list[dict[int, int|str]], batch_num: int) -> int:
    num_files_written = 0
    batch_dir = DESTINATION_DIR / f"batch{batch_num}"
    batch_dir.mkdir()
    mrns = set()
    for note in batch:
        # new patient encounted, create a patient directory
        if note["nc_pt_mrn"] not in mrns:
            (batch_dir / note["nc_pt_mrn"]).mkdir()
            mrns.add(note["nc_pt_mrn"])

        filename = batch_dir / note["nc_pt_mrn"] / f"{note['nc_pt_mrn']}_{note['nc_reportid']}_{note['sources']}.txt"
        filename.write_text(note["nc_clcont"])
        num_files_written += 1

    return num_files_written

def cancer_recur_json_to_text():
    note_dict: defaultdict[list[dict[int, int|str]]] = defaultdict(list)

    start = time.perf_counter()

    with CANCER_RECUR_JSON_FILE.open() as j_file:
        note_list = json.load(j_file)
        # reorganize into a dictionary by patient number
        for note in note_list:
#            print(note)
#            assert isinstance(note, dict)

            # make sure there is a note there, skip the blob only content
            if note.get("nc_clcont"):
                note_dict[note["nc_pt_mrn"]].append(note)

    DESTINATION_DIR.mkdir(exist_ok=True)


    NOTE_LIMIT = 700

    num_files_written = 0
    # write baches
    batch_num = 0
    batch = []
    for patient_notes in note_dict.values():

        if len(patient_notes) + len(batch) < NOTE_LIMIT:
            batch += patient_notes
        else:
            # write the batch to disk
            num_files_written += write_batch_to_disk(batch, batch_num)
            batch_num += 1
            batch = patient_notes

    # write the last batch
    if batch:
        num_files_written = write_batch_to_disk(batch, batch_num)
        batch_num += 1

    max_records = max([len(note_dict[key]) for key in note_dict])

#    print(f"{note_dict=}")
    print(f"Number of patients {len(note_dict)}")
    print(f"Number of records {sum([len(note_dict[key]) for key in note_dict])}")
    print(f"maximum records per patient {max_records}")
    print(" ---------- ")
    print(f"number of batches {batch_num}")
    print(f"number of files written {num_files_written}")
    print(f"took {time.perf_counter()-start}")


if __name__ == "__main__":
    cancer_recur_json_to_text()

