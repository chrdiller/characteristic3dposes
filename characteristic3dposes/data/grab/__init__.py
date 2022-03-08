from pathlib import Path
from typing import List, Tuple


def sample_ids(containing: str = None, excluding_actions: List[str] = []) -> List[str]:
    with open(str(Path(__file__).parent / 'index.txt'), 'r') as f:
        sample_ids = [line for line in f.read().replace('/', '-').split('\n')]
        if containing is not None:
            sample_ids = [sample_id for sample_id in sample_ids if containing in sample_id]
        sample_ids = [sample_id for sample_id in sample_ids if sample_id.split('-')[2] not in excluding_actions]
    return sample_ids


def subject_ids() -> List[str]:
    return [f's{idx}' for idx in range(1, 11)]


def sample_ids_for_subjects(subjects: List[str]) -> List[str]:
    output = []
    for subject in subjects:
        output.extend([sample_id for sample_id in sample_ids() if sample_id.startswith(f"{subject}-")])
    return output


def action_ids_for_subject(subject: str) -> List[str]:
    return ['-'.join(sample_id.split('-')[1:]) for sample_id in sample_ids() if sample_id.startswith(f"{subject}-")]


def split_sample_id(sample_id: str) -> Tuple[str, str]:
    subject = sample_id.split('-')[0]
    action_id = '-'.join(sample_id.split('-')[1:])
    return subject, action_id
