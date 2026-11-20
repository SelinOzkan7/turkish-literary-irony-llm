import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.data_loader import IronyDatasetLoader
from src.prompt_builder import PromptBuilder
from src.model_runner import LLMRunner
from src.evaluator import IronyEvaluator
from src.visualizer import ResultVisualizer

def run_experiment(dataset_path, models, conditions, strategies, output_dir="results"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    samples = IronyDatasetLoader(dataset_path).load()
    all_records, summary_records = [], []

    for model_name in models:
        short_model = model_name.split("/")[-1]
        runner = LLMRunner(model_name=model_name, load_in_4bit=True)
        for condition in conditions:
            for strategy in strategies:
                run_rows = []
                desc = f"{short_model} | {condition} | {strategy}"
                for sample in tqdm(samples, desc=desc):
                    prompt = PromptBuilder.build_prompt(sample, condition=condition, strategy=strategy)
                    max_toks = 450 if (strategy == "cot" or condition == "native_and_en") else 25
                    raw_out = runner.generate(prompt, max_new_tokens=max_toks)
                    pred = IronyEvaluator.extract_answer(raw_out)
                    row = {
                        "model": short_model, "condition": condition, "strategy": strategy,
                        "sample_id": sample.sample_id, "irony_type": sample.irony_type,
                        "ground_truth": sample.label, "predicted": pred, "raw_response": raw_out
                    }
                    run_rows.append(row)
                    all_records.append(row)

                metrics = IronyEvaluator.compute_metrics(pd.DataFrame(run_rows))
                metrics.update({"model": short_model, "condition": condition, "strategy": strategy, "model_strategy": f"{short_model} ({strategy})"})
                summary_records.append(metrics)
                print(f"Result [{desc}] -> Accuracy: {metrics['accuracy']}, F1: {metrics['macro_f1']}")

    pd.DataFrame(all_records).to_csv(out_path / "detailed_predictions.csv", index=False)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_path / "summary_metrics.csv", index=False)
    ResultVisualizer(str(out_path / "figures")).plot_condition_comparison(summary_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tr_lit_irony.jsonl")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    parser.add_argument("--conditions", nargs="+", default=["native_tr", "human_en", "mt_en", "native_and_en"])
    parser.add_argument("--strategies", nargs="+", default=["zero_shot", "cot"])
    args = parser.parse_args()
    run_experiment(args.data, args.models, args.conditions, args.strategies)
