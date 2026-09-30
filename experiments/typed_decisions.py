"""Typed Choice/Noul/Score adapter for a loaded Jev-Omni classifier.

The supplied classifier runs only where the caller loads it (Modal in this
project). This adapter preserves option keys and exposes raw model probabilities.
"""

import json


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                            sort_keys=True)


def evaluate_typed(classifier, request, model_id, *, media=None, modality="text"):
    """Answer a map of typed questions with one model call per question.

    Probabilities and confidence are raw classifier outputs. A downstream API
    should apply held-out calibration before calling them calibrated.
    """
    if not isinstance(request, dict) or "state" not in request:
        raise ValueError("Request must contain state")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Request must contain a nonempty questions map")
    if modality not in {"text", "image", "audio", "video"}:
        raise ValueError("Unsupported modality")
    if (modality == "text") != (media is None):
        raise ValueError("Media path and modality must agree")
    state = _text(request["state"])
    answers = {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id or not isinstance(question, dict):
            raise ValueError("Invalid question entry")
        kind = question.get("type")
        if kind not in {"noul", "choice", "score"} or "instructions" not in question:
            raise ValueError(f"Invalid question type or instructions: {question_id}")
        instructions = _text(question["instructions"])
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or
                                         set(criteria) - {"true", "false"}):
                raise ValueError("Noul criteria must have true/false keys")
            criteria = criteria or {}
            options = [f"false: {_text(criteria.get('false', 'No'))}",
                       f"true: {_text(criteria.get('true', 'Yes'))}"]
            keys = ["false", "true"]
        elif kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("Choice requires 2 to 255 criteria")
            keys = list(criteria)
            if any(not isinstance(key, str) or not key for key in keys):
                raise ValueError("Choice keys must be nonempty strings")
            options = [f"{key}: {_text(criteria[key])}" for key in keys]
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 ordered levels")
            keys = [str(index) for index in range(len(criteria))]
            options = [f"{index}: {_text(level)}" for index, level in enumerate(criteria)]
        if len(set(options)) != len(options):
            raise ValueError("Options must be distinct")
        result = classifier.predict(state=state, question=instructions,
                                    options=options, media=media, modality=modality)
        raw = result["probabilities"]
        if set(raw) != set(options):
            raise ValueError("Classifier returned an incomplete option distribution")
        probabilities = {key: float(raw[option]) for key, option in zip(keys, options)}
        if abs(sum(probabilities.values()) - 1.0) > 1e-3:
            raise ValueError("Classifier probabilities do not sum to one")
        if kind == "noul":
            answers[question_id] = {"type": kind, "noul": probabilities["true"]}
        elif kind == "choice":
            winner = max(keys, key=probabilities.get)
            answers[question_id] = {"type": kind, "choice": winner,
                                    "probabilities": probabilities,
                                    "confidence": float(result["confidence"])}
        else:
            answers[question_id] = {"type": kind,
                                    "score": sum(index * probabilities[str(index)]
                                                 for index in range(len(keys))),
                                    "legend": {str(index): _text(level)
                                               for index, level in enumerate(criteria)},
                                    "probabilities": probabilities,
                                    "confidence": float(result["confidence"])}
    return {"model": model_id, "answers": answers,
            "calibration_status": "raw_head_probabilities_unverified"}
