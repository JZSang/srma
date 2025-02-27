from dataclasses import dataclass
from typing import Optional, Union
from pydantic import BaseModel
from constants import TOKENIZER


class Item(BaseModel):
    abstract_in_investigation: Optional[str] = None
    abstract_in_investigation_actual_value: Optional[str] = None

    include_samples: int
    exclude_samples: int
    model: str = "gemini-pro"
    preprompt: str
    prompt: str

    gpt_cot_id: Optional[str] = None
    few_shot_exclude: int = 0
    few_shot_include: int = 0

    include_dataset: str
    exclude_dataset: str

    seed: int = 1
    ensemble: int = 1
    ensemble_threshold: Optional[int] = None

    is_batch: bool = False


class Result(BaseModel):
    prompt: str
    llm_answer: Union[str, None]

    correct: bool
    skipped: bool

    error: Union[str, None] = None
    predicted_value: Union[str, None] = ""
    predicted_values: list[str] = []
    actual_value: str = ""
    test_abstract: str = ""
    
    index_cao: str = "no_cao_index"

    token_counts: dict = {}


def zip_ensemble_abstracts(total_result_queue):
    from collections import defaultdict
    result_queue_dict = defaultdict(list)
    for result_raw in total_result_queue:
        result_queue_dict[result_raw["abstract_id"]].append(result_raw)

    return result_queue_dict


@dataclass
class FinalizeResultsReturn:
    total_correct: int
    total: int
    total_skipped: int

    results: list[Result]
    
    number_correct_includes: int
    number_correct_excludes: int
    number_completed_includes: int
    number_completed_excludes: int
    
    ensemble_threshold: int


def finalize_results(results, item: Item, ensemble_threshold) -> FinalizeResultsReturn:

    results_by_abstract_id = zip_ensemble_abstracts(results)
    
    correct_count = 0
    completed_count = 0
    skipped_count = 0
    final_results: list[Result] = []
    
    number_correct_includes = 0
    number_correct_excludes = 0
    number_completed_includes = 0
    number_completed_excludes = 0

    for res_key in results_by_abstract_id:
        from collections import Counter
        import tiktoken

        res_tuple = results_by_abstract_id[res_key]

        # Aggregate stats
        is_skipped = 0
        index_of_skipped = -1

        for i, res in enumerate(res_tuple):
            if res["skipped"]:
                is_skipped = 1
                index_of_skipped = i
                break
        skipped_count += is_skipped
        completed_count += not is_skipped

        # Token counting
        encoding = tiktoken.get_encoding(TOKENIZER)
        appended_llm_answers = [res.get("llm_answer") or "" for res in res_tuple]
        llm_answers_token_counts = sum(
            [
                len(encoding.encode(appended_llm_answer))
                for appended_llm_answer in appended_llm_answers
            ]
        )
        prompt_token_count = len(encoding.encode(res["prompt"]))
        token_counts = {
            "llm_answer": llm_answers_token_counts,
            "prompt": prompt_token_count,
            "prompt_total": prompt_token_count * item.ensemble,
        }

        number_of_predictions_for_included = Counter(
            [res["predicted_value"] for res in res_tuple]
        )["included"]
        is_included = number_of_predictions_for_included >= ensemble_threshold

        prompt = res_tuple[0]["prompt"]
        llm_answer = "\n---\n".join(appended_llm_answers)
        skipped = True if is_skipped else False
        predicted_value = "included" if is_included else "excluded"
        actual_value = "fail" if is_skipped else res_tuple[0]["actual_value"]
        correct = predicted_value == actual_value
        error = str(res_tuple[index_of_skipped]["error"]) if is_skipped else None
        test_abstract = res_tuple[0]["test_abstract"]
        token_counts = token_counts
        predicted_values = [res["predicted_value"] for res in res_tuple]
        index_cao = res_tuple[0].get("index_cao", "no_cao_index")

        correct_count += correct if not is_skipped else 0
        if actual_value == "included":
            number_completed_includes += 1
            if correct:
                number_correct_includes += 1
        elif actual_value == "excluded":
            number_completed_excludes += 1
            if correct:
                number_correct_excludes += 1

        final_results.append(
            Result(
                prompt=prompt,
                llm_answer=llm_answer,
                correct=correct,
                skipped=skipped,
                predicted_value=predicted_value,
                actual_value=actual_value,
                error=error,
                test_abstract=test_abstract,
                token_counts=token_counts,
                predicted_values=predicted_values,
                index_cao=str(index_cao),
            )
        )
    
    return FinalizeResultsReturn(
        total_correct=correct_count,
        total=completed_count,
        total_skipped=skipped_count,
        results=final_results,
        ensemble_threshold=ensemble_threshold,
        number_correct_includes=number_correct_includes,
        number_correct_excludes=number_correct_excludes,
        number_completed_includes=number_completed_includes,
        number_completed_excludes=number_completed_excludes,
    )
