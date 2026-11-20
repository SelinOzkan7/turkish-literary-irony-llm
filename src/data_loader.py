import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class IronySample:
    sample_id: str
    author: str
    work: str
    irony_type: str
    text_tr: str
    text_en_human: str
    text_en_mt: str
    option_a_tr: str
    option_b_tr: str
    option_a_en: str
    option_b_en: str
    label: str

class IronyDatasetLoader:
    VALID_LABELS = {"A", "B"}
    VALID_TYPES = {"verbal_irony", "situational_satire", "literal_control"}

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.samples: List[IronySample] = []

    def load(self) -> List[IronySample]:
        loaded_samples = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    loaded_samples.append(IronySample(**json.loads(line)))
        self.samples = loaded_samples
        return self.samples

    def get_summary(self) -> Dict[str, int]:
        summary = {"total": len(self.samples)}
        for t in self.VALID_TYPES:
            summary[t] = sum(1 for s in self.samples if s.irony_type == t)
        return summary
