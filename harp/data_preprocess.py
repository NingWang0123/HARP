import string
from typing import List, Dict, Any
import json
import numpy as np

def mmlu_to_sft(ex: Dict[str, Any], include_subject: bool = True, mode: str = "eval") -> Dict[str, str]:
    """
    ex has keys: question, choices (list[str]), answer (int index)
    Optionally may have: subject
    """
    q = str(ex.get("question", "")).strip()
    choices = ex.get("choices", [])
    if not isinstance(choices, list):
        # some variants store as tuple or other types
        choices = list(choices) if choices is not None else []

    # build options as A. xxx
    letters = list(string.ascii_uppercase)
    opt_lines: List[str] = []
    for i, c in enumerate(choices):
        if i >= len(letters):
            break
        opt_lines.append(f"{letters[i]}. {str(c).strip()}")

    subject = str(ex.get("subject", "")).strip()
    # header = f"Subject: {subject}\n" if (include_subject and subject) else ""

    prompt = (
        f"The following is a multiple-choice question. Please choose the correct answer.\n\n"
        f"Question: {ex['question']}\n"
        f"A. {ex['choices'][0]}\n"
        f"B. {ex['choices'][1]}\n"
        f"C. {ex['choices'][2]}\n"
        f"D. {ex['choices'][3]}\n"
        f"### Answer:"
    ).strip()

    ans = ex.get("answer", None)
    # answer is usually int index
    if isinstance(ans, (int, np.integer)):
        if 0 <= int(ans) < len(opt_lines) and int(ans) < len(letters):
            target = letters[int(ans)]
        else:
            target = ""
    else:
        # some variants store as "A"/"B"/...
        target = str(ans).strip()
    target = "Based on the choices above, the answer is " + target + "." + f" {ex['choices'][ans]}" if mode == "train" else target
    
    return {"input": prompt, "target": target}

def arc_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    ex has keys:
      - question (str)
      - choices (dict with keys 'text' (list) and 'label' (list))
      - answerKey (str, e.g., 'A', '1', 'B')
    """
    question = str(ex.get("question", "")).strip()
    
    # ARC 'choices' is often a dict: {'text': ['...','...'], 'label': ['A','B']}
    choices_dict = ex.get("choices", {})
    choice_texts = choices_dict.get("text", [])
    choice_labels = choices_dict.get("label", [])
    
    # Build the options string (e.g., "A. xxxx")
    # Note: ARC labels are usually A,B,C,D or 1,2,3,4. We trust the dataset's label or normalize it.
    opt_lines = []
    
    # We maintain a mapping to find the answer text later if needed
    label_to_text = {} 
    
    for label, text in zip(choice_labels, choice_texts):
        label_str = str(label).strip()
        text_str = str(text).strip()
        opt_lines.append(f"{label_str}. {text_str}")
        label_to_text[label_str] = text_str

    options_str = "\n".join(opt_lines)

    prompt = (
        f"The following is a multiple-choice question. Please choose the correct answer.\n\n"
        f"Question: {question}\n"
        f"{options_str}\n"
        f"### Answer:"
    ).strip()

    # Handle the target/answer
    answer_key = str(ex.get("answerKey", "")).strip()
    
    # If mode is "train", we create a verbose answer; otherwise, just the key.
    if mode == "train":
        # Retrieve the text corresponding to the answer key
        answer_text = label_to_text.get(answer_key, "")
        target = f"Based on the choices above, the answer is {answer_key}. {answer_text}"
    else:
        target = answer_key

    return {"input": prompt, "target": target}

def ifeval_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    IFEval (HuggingFaceH4/ifeval): instruction following with verifiable constraints.
    keys: key (int), prompt (str), instruction_id_list (list), kwargs (list of dicts)

    No canonical "correct response" — final eval scores via constraint-checks.
    For HARP Stage 1 inner-eval we use the prompt itself as a self-similarity
    reference (a model that addresses the prompt's topics will lexically overlap
    with it). Crude, but at least non-zero and sample-discriminative.
    """
    prompt_text = str(ex.get("prompt", "")).strip()

    if mode == "train":
        target = " " + str(ex.get("response", "")).strip()
    else:
        target = prompt_text  # weak signal but non-zero
    return {"input": prompt_text, "target": target}


def humaneval_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    HumanEval: code completion. Just feed the function-signature prompt directly.
    keys: task_id, prompt (signature + docstring), canonical_solution, test, entry_point
    """
    prompt_text = str(ex.get("prompt", ""))
    # Use canonical_solution as gold in BOTH modes. Final eval (vllm_eval_humaneval)
    # actually runs unit tests, but HARP Stage 1 inner-eval needs a token-level
    # reference signal — canonical_solution is the obvious choice.
    target = str(ex.get("canonical_solution", ""))
    return {"input": prompt_text, "target": target}


def gsm8k_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    GSM8K: math word problems with chain-of-thought.
    keys: question (str), answer (str — full reasoning ending with "#### <number>")
    Returns prompt asking for step-by-step reasoning, target = numeric answer string.
    """
    import re
    question = str(ex.get("question", "")).strip()
    answer_full = str(ex.get("answer", "")).strip()

    # Extract gold numeric answer after the "#### " marker
    m = re.search(r"####\s*(.+)$", answer_full)
    gold_num = m.group(1).strip().replace(",", "") if m else ""

    prompt = (
        f"Solve the following math word problem step by step. Provide the final numeric answer after '#### '.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    ).strip()

    if mode == "train":
        target = " " + answer_full  # full reasoning chain
    else:
        target = gold_num  # just the number for eval comparison
    return {"input": prompt, "target": target}


def math500_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    MATH-500: competition math, LaTeX answer wrapped in \\boxed{}.
    keys: problem (str), solution (str), answer (str — boxed answer extracted), level, subject
    """
    problem = str(ex.get("problem", "")).strip()
    answer = str(ex.get("answer", "")).strip()

    prompt = (
        f"Solve the following math problem. Wrap the final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}\n"
        f"Solution:"
    ).strip()

    if mode == "train":
        solution = str(ex.get("solution", "")).strip()
        target = " " + solution
    else:
        target = answer
    return {"input": prompt, "target": target}


def squad_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    SQuAD v1: extractive QA — passage + question → span answer.
    keys: context (str), question (str), answers (dict: text [list of accepted str])
    """
    context = str(ex.get("context", "")).strip()
    question = str(ex.get("question", "")).strip()
    answers = ex.get("answers", {})
    answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
    gold = answer_texts[0].strip() if answer_texts else ""

    prompt = (
        f"Read the following passage and answer the question with a short span from the passage.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    ).strip()

    if mode == "train":
        target = " " + gold
    else:
        # Return the FIRST gold answer; eval will compare against all valid answers via metric_squad
        target = gold if gold else ""
    return {"input": prompt, "target": target}


def piqa_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    PIQA: physical commonsense reasoning, 2 choices.
    keys: goal (str), sol1 (str), sol2 (str), label (int 0 or 1)
    """
    goal = str(ex.get("goal", "")).strip()
    sol1 = str(ex.get("sol1", "")).strip()
    sol2 = str(ex.get("sol2", "")).strip()
    label = int(ex.get("label", -1))
    target_letter = "A" if label == 0 else "B" if label == 1 else ""

    prompt = (
        f"The following is a multiple-choice question. Please choose the correct answer.\n\n"
        f"Question: {goal}\n"
        f"A. {sol1}\n"
        f"B. {sol2}\n"
        f"### Answer:"
    ).strip()

    if mode == "train" and target_letter:
        chosen = sol1 if target_letter == "A" else sol2
        target = f"Based on the choices above, the answer is {target_letter}. {chosen}"
    else:
        target = target_letter
    return {"input": prompt, "target": target}


def truthfulqa_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    TruthfulQA mc1 (single correct answer among variable-length choices).
    keys: question (str), mc1_targets (dict: choices [list of str], labels [list of int, exactly one is 1])
    """
    import string
    question = str(ex.get("question", "")).strip()
    mc1 = ex.get("mc1_targets", {})
    choices = list(mc1.get("choices", []))
    labels = list(mc1.get("labels", []))

    letters = list(string.ascii_uppercase)
    opt_lines = []
    target_letter = ""
    for i, (choice, label) in enumerate(zip(choices, labels)):
        if i >= len(letters):
            break
        opt_lines.append(f"{letters[i]}. {str(choice).strip()}")
        if int(label) == 1 and not target_letter:
            target_letter = letters[i]

    prompt = (
        f"The following is a multiple-choice question. Please choose the correct answer.\n\n"
        f"Question: {question}\n"
        + "\n".join(opt_lines) + "\n"
        f"### Answer:"
    ).strip()

    if mode == "train" and target_letter:
        idx = letters.index(target_letter)
        target = f"Based on the choices above, the answer is {target_letter}. {choices[idx]}"
    else:
        target = target_letter
    return {"input": prompt, "target": target}


def mathqa_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    MathQA: math word problems with 5 choices (a-e).
    keys: Problem (str), options (str like "a ) X , b ) Y , c ) Z , d ) W , e ) V"), correct (str: letter a-e)
    """
    import re
    problem = str(ex.get("Problem", "")).strip()
    options_str = str(ex.get("options", "")).strip()
    correct = str(ex.get("correct", "")).strip().lower()

    # Parse "a ) ... , b ) ... , c ) ..." into individual choice texts
    pieces = re.split(r"\s*,\s*(?=[a-e]\s*\))", options_str)
    choice_map = {}
    for piece in pieces:
        m = re.match(r"\s*([a-e])\s*\)\s*(.*)", piece, re.IGNORECASE)
        if m:
            letter = m.group(1).lower()
            text = m.group(2).strip()
            choice_map[letter] = text

    # Build options A-E (uppercase to match prompt style)
    letters_lower = ["a", "b", "c", "d", "e"]
    opt_lines = []
    for i, lower_letter in enumerate(letters_lower):
        if lower_letter in choice_map:
            upper_letter = lower_letter.upper()
            opt_lines.append(f"{upper_letter}. {choice_map[lower_letter]}")

    target_letter = correct.upper() if correct in letters_lower else ""

    prompt = (
        f"The following is a multiple-choice question. Please choose the correct answer.\n\n"
        f"Question: {problem}\n"
        + "\n".join(opt_lines) + "\n"
        f"### Answer:"
    ).strip()

    if mode == "train" and target_letter:
        target = f"Based on the choices above, the answer is {target_letter}. {choice_map.get(correct, '')}"
    else:
        target = target_letter
    return {"input": prompt, "target": target}


def wizardlm_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    Extracts 'instruction' and 'output' from the example.
    Returns empty strings if keys are missing.

    Format chosen to match openmath_to_sft / magicoder_to_sft style:
    target gets a leading separator (single space) but no role marker.
    Avoids forcing the model to learn an Alpaca-style "### Response:" pattern
    that doesn't transfer to MCQ evals like MMLU/ARC.
    """
    prompt = str(ex.get("instruction", "")).strip()
    target = str(ex.get("output", "")).strip()
    if mode == "train":
        target = " " + target
    return {"input": prompt, "target": target}

def self_instruct_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    yizhongw/self_instruct (config=self_instruct) has top-level {prompt, completion}.
    Per the original paper, ~46% of samples have problems (52% noise rate on
    input-output pairs). Used here as a deliberately-noisy SFT pool for
    differentiating data-selection methods.
    """
    prompt = str(ex.get("prompt", "")).strip()
    target = str(ex.get("completion", "")).strip()
    if mode == "train":
        target = " " + target
    return {"input": prompt, "target": target}

def alpaca_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    Extracts 'instruction' and 'output' from the example.
    Returns empty strings if keys are missing.
    """
    instruction = str(ex.get("instruction", "")).strip()
    input = str(ex.get("input", "")).strip()
    if input:
        prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                  f"### Instruction:{instruction}\n\n"
                  f"### Input:{input}\n\n"
                  f"### Response:").strip()
    else:
        prompt = (f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                  f"### Instruction:{instruction}\n"
                  f"### Response:").strip()

    target = str(ex.get("output", "")).strip()
    return {"input": prompt, "target": target}


def openmath_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    NVIDIA OpenMathInstruct-2 row format:
      problem (str): math problem statement
      generated_solution (str): full chain-of-thought solution
      expected_answer (str): final concise answer (e.g., "45", "\\frac{7}{12}")
      problem_source (str): one of {gsm8k, math, augmented_gsm8k, augmented_math}

    For training: prompt = problem, completion = full chain-of-thought solution.
    """
    problem = str(ex.get("problem", "")).strip()
    solution = str(ex.get("generated_solution", "")).strip()
    prompt = (
        f"Below is a math problem. Solve it step by step and put the final answer "
        f"in \\boxed{{}}.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution:"
    ).strip()
    if mode == "train":
        target = " " + solution
    else:
        target = str(ex.get("expected_answer", "")).strip()
    return {"input": prompt, "target": target}


def magicoder_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    Magicoder-OSS-Instruct-75K row format:
      lang (str): programming language (e.g., 'python', 'cpp', ...)
      problem (str): problem description
      solution (str): code solution
      seed (str): original instruction seed
      raw_index, index, openai_fingerprint: metadata (ignored)

    For training: prompt = problem, completion = solution.
    """
    problem = str(ex.get("problem", "")).strip()
    solution = str(ex.get("solution", "")).strip()
    prompt = (
        f"Below is a coding problem. Write a complete solution.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution:"
    ).strip()
    if mode == "train":
        target = "\n" + solution
    else:
        target = solution
    return {"input": prompt, "target": target}

def tulu3_to_sft(ex: Dict[str, Any], mode: str = "eval") -> Dict[str, str]:
    """
    allenai/tulu-3-sft-mixture row format:
      messages (list[{"role": "user"|"assistant"|"system", "content": str}])
      id, source, …  (metadata; ignored)

    Multi-turn dialogs are flattened: the prompt accumulates all user/system
    turns up to (but not including) the final assistant turn, and the final
    assistant turn becomes the target. Single-turn rows (the majority) fall
    through cleanly. Rows without the {role,content} schema return empty
    strings so the caller can skip / fall back.

    Format aligns with openmath_to_sft / magicoder_to_sft (separator-prefixed
    target in train mode, no Alpaca-style role markers).
    """
    messages = ex.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 2:
        return {"input": "", "target": ""}

    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx <= 0:
        return {"input": "", "target": ""}

    target = str(messages[last_assistant_idx].get("content", "")).strip()
    prompt_parts = []
    for m in messages[:last_assistant_idx]:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt = "\n\n".join(prompt_parts).strip()

    if mode == "train":
        target = " " + target
    return {"input": prompt, "target": target}


def convert_dataset_to_sft(examples: List[Dict[str, Any]], mode: str = "eval", dataset_name: str = "mmlu") -> List[Dict[str, str]]:
    out = []
    for ex in examples:
        if dataset_name == "mmlu":
            sft = mmlu_to_sft(ex, mode=mode)
        elif dataset_name == "arc":
            sft = arc_to_sft(ex, mode=mode)
        elif dataset_name == "wizardlm":
            sft = wizardlm_to_sft(ex, mode=mode)
        elif dataset_name == "alpaca":
            sft = alpaca_to_sft(ex, mode=mode)
        elif dataset_name == "openmath":
            sft = openmath_to_sft(ex, mode=mode)
        elif dataset_name == "magicoder":
            sft = magicoder_to_sft(ex, mode=mode)
        elif dataset_name == "tulu3":
            sft = tulu3_to_sft(ex, mode=mode)
        elif dataset_name == "self_instruct":
            sft = self_instruct_to_sft(ex, mode=mode)
        elif dataset_name == "squad":
            sft = squad_to_sft(ex, mode=mode)
            # Carry through ALL gold answer texts so eval can compute max F1
            ans = ex.get("answers", {})
            if isinstance(ans, dict):
                sft["answers"] = ans.get("text", []) or []
            else:
                sft["answers"] = []
        elif dataset_name == "gsm8k":
            sft = gsm8k_to_sft(ex, mode=mode)
        elif dataset_name == "math":
            sft = math500_to_sft(ex, mode=mode)
        elif dataset_name == "piqa":
            sft = piqa_to_sft(ex, mode=mode)
        elif dataset_name == "truthfulqa":
            sft = truthfulqa_to_sft(ex, mode=mode)
        elif dataset_name == "mathqa":
            sft = mathqa_to_sft(ex, mode=mode)
        elif dataset_name == "humaneval":
            sft = humaneval_to_sft(ex, mode=mode)
        elif dataset_name == "ifeval":
            sft = ifeval_to_sft(ex, mode=mode)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        if not sft["input"]:
            sft["input"] = json.dumps(ex, ensure_ascii=False)
        if sft["target"] is None:
            sft["target"] = ""
        subject = str(ex.get("subject", "1")).strip()
        sft["subject"] = subject
        # Track task_type for downstream scoring (multi_choice / numeric / latex_boxed / extractive)
        task_type_map = {
            "mmlu": "multi_choice", "arc": "multi_choice",
            "piqa": "multi_choice", "truthfulqa": "multi_choice", "mathqa": "multi_choice",
            "gsm8k": "numeric", "math": "latex_boxed",
            "squad": "extractive",
            "wizardlm": "generation", "alpaca": "generation",
            "openmath": "generation", "magicoder": "generation",
            "self_instruct": "generation",
            "humaneval": "code", "ifeval": "generation",
        }
        sft["task_type"] = task_type_map.get(dataset_name, "multi_choice")
        out.append(sft)
    return out


# ---------- Task-type-aware scoring (used by HARP Stage 1a/1b accuracy) ----------

def _normalize_squad(s: str) -> str:
    """SQuAD normalization: lowercase, strip articles, punctuation, extra whitespace."""
    import re as _re
    s = s.lower()
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def score_response(task_type: str, pred_text: str, gold, **kwargs) -> float:
    """
    Compute a 0-1 score for a model response, given task type and gold answer.

    Args:
      task_type: one of "multi_choice", "numeric", "latex_boxed", "extractive", "generation"
      pred_text: raw model output (string)
      gold: gold answer (str, or list of strs for "extractive")
      kwargs: optional — e.g. max_letter for multi_choice

    Returns:
      Float in [0, 1]. For multi_choice/numeric/latex_boxed it's 0 or 1; for
      extractive it's max F1 over gold answers.
    """
    import re as _re
    pred = str(pred_text).strip()

    if task_type == "multi_choice":
        max_letter = kwargs.get("max_letter", "D")
        gold_char = str(gold).strip().upper()
        if not gold_char:
            return 0.0
        rng = f"[A-{max_letter}]"
        m = _re.search(rf"(?:answer|option)(?:\s+is)?\s*[:\s]\s*({rng})", pred, _re.IGNORECASE)
        if m:
            return 1.0 if m.group(1).upper() == gold_char else 0.0
        m = _re.search(rf"^\s*({rng})(?:\W|$)", pred, _re.IGNORECASE)
        if m:
            return 1.0 if m.group(1).upper() == gold_char else 0.0
        return 0.0

    if task_type == "numeric":
        m = _re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", pred)
        if m:
            cand = m.group(1).replace(",", "").strip()
        else:
            nums = _re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
            cand = nums[-1] if nums else ""
        gold_str = str(gold).replace(",", "").strip()
        if not cand or not gold_str:
            return 0.0
        try:
            return 1.0 if abs(float(cand) - float(gold_str)) < 1e-3 else 0.0
        except ValueError:
            return 1.0 if cand == gold_str else 0.0

    if task_type == "latex_boxed":
        ms = _re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", pred)
        if ms:
            cand = ms[-1].strip()
        else:
            m = _re.search(r"answer\s+is\s*[:\s]\s*([^\n.]+)", pred, _re.IGNORECASE)
            cand = m.group(1).strip().rstrip('.').strip('$').strip() if m else ""

        def _norm(s):
            s = s.strip().rstrip('.').strip()
            s = s.replace(" ", "").replace("\\!", "").replace("\\,", "")
            s = s.replace("\\left", "").replace("\\right", "")
            return s.lower()

        if not cand or not gold:
            return 0.0
        return 1.0 if _norm(cand) == _norm(str(gold)) else 0.0

    if task_type == "extractive":
        # gold is expected to be a list of accepted answer strings (carry over from convert_dataset_to_sft)
        if isinstance(gold, str):
            gold_list = [gold] if gold else []
        else:
            gold_list = [g for g in gold if g]
        if not gold_list:
            return 0.0
        from collections import Counter as _Counter
        pred_tokens = _normalize_squad(pred).split()
        if not pred_tokens:
            return 0.0
        best = 0.0
        for g in gold_list:
            gt = _normalize_squad(g).split()
            if not gt:
                continue
            common = _Counter(pred_tokens) & _Counter(gt)
            same = sum(common.values())
            if same == 0:
                continue
            p = same / len(pred_tokens)
            r = same / len(gt)
            f1 = 2 * p * r / (p + r)
            best = max(best, f1)
        return best

    if task_type in ("generation", "code"):
        # No canonical-answer scoring; use token-level F1 vs gold as a proxy.
        # Gives HARP/SHED Stage 1 a non-zero, sample-discriminative signal so
        # Shapley estimation can distinguish helpful from unhelpful rows.
        gold_str = "" if gold is None else (gold if isinstance(gold, str) else str(gold))
        gold_str = gold_str.strip()
        if not gold_str:
            return 0.0
        from collections import Counter as _Counter
        # Lowercase + simple word-tokenize
        pred_tokens = _re.findall(r"\w+", pred.lower())
        gold_tokens = _re.findall(r"\w+", gold_str.lower())
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = _Counter(pred_tokens) & _Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            return 0.0
        p = same / len(pred_tokens)
        r = same / len(gold_tokens)
        return 2 * p * r / (p + r)

    # unknown — no auto-scoring
    return 0.0

