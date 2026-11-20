from typing import Literal
from src.data_loader import IronySample

ConditionType = Literal["native_tr", "human_en", "mt_en", "native_and_en"]
StrategyType = Literal["zero_shot", "cot"]

class PromptBuilder:
    @staticmethod
    def build_prompt(sample: IronySample, condition: ConditionType = "native_tr", strategy: StrategyType = "zero_shot") -> str:
        if condition == "native_tr":
            passage, opt_a, opt_b, lang_desc = sample.text_tr, sample.option_a_tr, sample.option_b_tr, "Turkish"
        elif condition == "human_en":
            passage, opt_a, opt_b, lang_desc = sample.text_en_human, sample.option_a_en, sample.option_b_en, "English (translated from Turkish literature)"
        elif condition == "mt_en":
            passage, opt_a, opt_b, lang_desc = sample.text_en_mt, sample.option_a_en, sample.option_b_en, "English (machine-translated from Turkish literature)"
        elif condition == "native_and_en":
            return PromptBuilder._build_xlt_prompt(sample, strategy)
        else:
            raise ValueError(f"Unsupported condition: {condition}")

        base = (
            f"Question: How should the speaker's or narrator's intended meaning in the following "
            f"{lang_desc} literary passage be interpreted? Consider context, tone, and potential irony.\n"
            f"Passage: \"{passage}\"\nChoices:\nA: {opt_a}\nB: {opt_b}\n"
        )
        if strategy == "zero_shot":
            return base + "Reply ONLY with 'Answer: A' or 'Answer: B'.\nAnswer:"
        return base + "In 2-3 concise sentences, compare the literal meaning versus any contextual/ironic tone, and strictly conclude with 'Answer: A' or 'Answer: B'.\nExplanation:"

    @staticmethod
    def _build_xlt_prompt(sample: IronySample, strategy: StrategyType) -> str:
        prompt = (
            "Question: How should the intended meaning in this Turkish literary passage be interpreted? "
            "First, translate the passage and choices into English to capture the pragmatic nuance, then choose between A and B.\n"
            f"Passage (Turkish): \"{sample.text_tr}\"\nChoices:\nA: {sample.option_a_tr}\nB: {sample.option_b_tr}\n"
        )
        return prompt + "Provide your English translation, a brief analysis of any irony, and finish with 'Answer: A' or 'Answer: B'.\nResponse:"
