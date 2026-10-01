"""Ollama/TypeSafe wire adapter preserving Glim's native decision head.

This is a protocol adapter, not a stock Ollama model runner. No model is loaded
by importing it. Callers run inference remotely in this project.
"""
import math


def evaluate_systemone(classifier, request, model_id):
    from experiments.typed_decisions import evaluate_typed
    if not isinstance(request, dict) or request.get("model") != model_id:
        raise ValueError("model must match the loaded model")
    state = request.get("state")
    if not isinstance(state, (str, dict, list)) or (isinstance(state, str) and not state.strip()):
        raise ValueError("state must be a nonempty string, object or array")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("questions must contain 1 to 64 fields")
    normalized = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(question, dict):
            raise ValueError("invalid named question")
        instructions = question.get("instructions")
        if not isinstance(instructions, (str, dict, list)) or (isinstance(instructions, str) and not instructions.strip()):
            raise ValueError("instructions must be a nonempty string, object or array")
        kind = question.get("type"); criteria = question.get("criteria")
        item = {"type": kind, "instructions": instructions}
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
                raise ValueError("choice requires 2 to 26 candidates")
            if any(not isinstance(k, str) or not k.strip() or (v is not None and not isinstance(v, str)) for k,v in criteria.items()):
                raise ValueError("choice criteria must map nonempty keys to strings or null")
            item["criteria"] = {k:k if v is None else v for k,v in criteria.items()}
        elif kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria)-{"true","false"} or any(not isinstance(v,str) for v in criteria.values()):
                    raise ValueError("noul criteria must use true/false string descriptions")
                item["criteria"] = criteria
        elif kind == "score":
            if not isinstance(criteria,list) or not 2 <= len(criteria) <= 10 or any(not isinstance(v,str) for v in criteria):
                raise ValueError("Glim score requires 2 to 10 ordered string levels")
            item["criteria"] = criteria
        else:
            raise ValueError("type must be choice, noul or score")
        normalized[name] = item
    result = evaluate_typed(classifier,{"state":state,"questions":normalized},model_id)
    for answer in result["answers"].values():
        if answer["type"] == "noul":
            if not math.isfinite(answer["noul"]) or not 0 <= answer["noul"] <= 1:
                raise ValueError("invalid native Noul probability")
            continue
        probabilities = list(answer["probabilities"].values())
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities)-1)>1e-3:
            raise ValueError("invalid native probability distribution")
        entropy = -sum(p*math.log(p) for p in probabilities if p)
        answer["confidence"] = max(0.,min(1.,1-entropy/math.log(len(probabilities))))
    # Token counts are not exposed by the native classifier. Do not invent usage.
    return {"model": model_id, "answers": result["answers"]}
