import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def populate_machine_translations(input_path="data/tr_lit_irony.jsonl", output_path="data/tr_lit_irony.jsonl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading NLLB-200 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="tur_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    forced_bos_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    for rec in tqdm(records, desc="Translating TR -> EN"):
        if not rec.get("text_en_mt"):
            inputs = tokenizer(rec["text_tr"], return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, forced_bos_token_id=forced_bos_id, max_new_tokens=256)
            rec["text_en_mt"] = tokenizer.decode(gen[0], skip_special_tokens=True).strip()

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Machine translations populated successfully!")

if __name__ == "__main__":
    populate_machine_translations()
