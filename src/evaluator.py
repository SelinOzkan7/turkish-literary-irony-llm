import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class IronyEvaluator:
    @staticmethod
    def extract_answer(raw_response: str) -> str:
        match = re.search(r"Answer\s*:\s*([AB])\b", raw_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        fallback = re.findall(r"\b([AB])\b", raw_response)
        return fallback[-1].upper() if fallback else "INVALID"

    @staticmethod
    def compute_metrics(df: pd.DataFrame) -> dict:
        valid_df = df[df["predicted"].isin(["A", "B"])]
        y_true = df["ground_truth"].tolist()
        y_pred = [p if p in ["A", "B"] else ("B" if g == "A" else "A") for p, g in zip(df["predicted"], y_true)]
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "parse_rate": round(len(valid_df) / len(df), 4),
            "total_samples": len(df),
        }
