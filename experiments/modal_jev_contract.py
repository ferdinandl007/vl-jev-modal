"""Jev-shaped zero-training image/video decisions. Inference and media stay on Modal.

This is an experimental compatible wire shape, not a Jev model or a calibrated
replacement for TypeSafe's unpublished confidence calculation.
"""

import json
import math
import string

import modal


app = modal.App("vl-jev-contract-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.10.2",
        "accelerate==1.13.0",
        "av==16.1.0",
        "pillow==12.1.1",
        "requests==2.32.5",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
cache = modal.Volume.from_name("vl-jev-zero-shot-hf-cache", create_if_missing=True)
probe_media = modal.Volume.from_name("vl-jev-probe-media", create_if_missing=True)
MODEL_NAME = "Qwen/Qwen3.5-2B"
API_MODEL_NAME = "vl-jev-qwen3.5-2b-zero-shot"


def concentration_confidence(probabilities):
    """Provisional entropy proxy; TypeSafe does not publish its exact formula."""
    if len(probabilities) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probabilities))))


def question_candidates(question):
    if not isinstance(question, dict):
        raise ValueError("Each question must be an object")
    kind = question.get("type")
    if not isinstance(question.get("instructions"), (str, dict, list, type(None))):
        raise ValueError("instructions must be text, an object, an array, or null")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("choice criteria must contain 1 to 255 named options")
        if any(not isinstance(name, str) or not name for name in criteria):
            raise ValueError("choice names must be nonempty strings")
        if any(not isinstance(value, (str, dict, list, type(None))) for value in criteria.values()):
            raise ValueError("choice descriptions must be text, object, array, or null")
        return list(criteria), [
            name if detail is None else f"{name}: {json.dumps(detail, ensure_ascii=False)}"
            for name, detail in criteria.items()
        ]
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must contain 2 to 10 ordered levels")
        if any(not isinstance(value, (str, dict, list)) for value in criteria):
            raise ValueError("score levels must be text, objects, or arrays")
        return [str(i) for i in range(len(criteria))], [
            json.dumps(level, ensure_ascii=False) for level in criteria
        ]
    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an optional object")
        if any(not isinstance(criteria.get(side), (str, dict, list, type(None))) for side in ("true", "false")):
            raise ValueError("noul criteria must be text, objects, arrays, or null")
        negative = criteria.get("false", "No; the statement is false")
        positive = criteria.get("true", "Yes; the statement is true")
        return ["false", "true"], [
            json.dumps(negative, ensure_ascii=False),
            json.dumps(positive, ensure_ascii=False),
        ]
    raise ValueError(f"Unknown question type: {kind!r}")


def make_answer(question, keys, probabilities):
    if len(keys) != len(probabilities) or any(
        not math.isfinite(p) or p < 0 or p > 1 for p in probabilities
    ) or abs(sum(probabilities) - 1.0) > 1e-4:
        raise ValueError("Decoder returned an invalid probability distribution")
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    distribution = dict(zip(keys, probabilities))
    confidence = concentration_confidence(probabilities)
    if kind == "choice":
        winner = keys[max(range(len(keys)), key=lambda i: probabilities[i])]
        return {
            "type": "choice",
            "choice": winner,
            "confidence": confidence,
            "probabilities": distribution,
        }
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probabilities)),
        "confidence": confidence,
        "legend": dict(zip(keys, question["criteria"])),
        "probabilities": distribution,
    }


def question_messages(media, state_context, question, labels):
    keys, descriptions = question_candidates(question)
    prompt = (
        "Decide using only the supplied state and visual evidence. "
        "Return the single label of the best answer.\n"
        f"State: {json.dumps(state_context, ensure_ascii=False)}\n"
        f"Question: {json.dumps(question.get('instructions', ''), ensure_ascii=False)}\n"
        + "\n".join(
            f"{label}. {description}"
            for label, description in zip(labels[: len(keys)], descriptions)
        )
        + "\nAnswer: "
    )
    content = ([media] if media else []) + [{"type": "text", "text": prompt}]
    return keys, [{"role": "user", "content": content}]


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/model-cache": cache, "/probe-media": probe_media},
    timeout=1800,
)
class DecisionModel:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        torch.set_grad_enabled(False)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, max_image_size={"longest_edge": 50176}
        )
        self.processor.tokenizer.padding_side = "right"
        self.processor.video_processor.fps = None
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        token_ids = []
        labels = []
        candidates = (
            list(string.digits)
            + list(string.ascii_uppercase)
            + list(string.ascii_lowercase)
            + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
            + [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
        )
        for label in candidates:
            ids = self.processor.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1 or ids[0] in token_ids:
                continue
            if self.processor.tokenizer.decode(ids) != label:
                continue
            labels.append(label)
            token_ids.append(ids[0])
            if len(labels) == 255:
                break
        if len(labels) != 255:
            raise ValueError(f"Only {len(labels)} unambiguous one-token labels available")
        self.labels = labels
        self.label_token_ids = torch.tensor(token_ids, device="cuda", dtype=torch.long)
        self.label_weights = self.model.lm_head.weight.index_select(
            0, self.label_token_ids
        ).detach()
        cache.commit()

    def media_from_state(self, state):
        from pathlib import Path
        from urllib.parse import urlparse

        import requests

        if not isinstance(state, dict) or "media" not in state:
            return None, state
        media = state["media"]
        if not isinstance(media, dict) or media.get("type") not in ("image", "video"):
            raise ValueError("state.media needs type=image or type=video")
        url = media.get("url")
        if not isinstance(url, str):
            raise ValueError("state.media.url must be a string")
        if url.startswith("/probe-media/"):
            path = Path(url)
            if not path.is_file():
                raise ValueError("Remote test media is missing")
        else:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("Only HTTPS media URLs are accepted")
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            if len(response.content) > 50_000_000:
                raise ValueError("Media exceeds the 50 MB experiment limit")
            suffix = ".mp4" if media["type"] == "video" else ".jpg"
            path = Path("/tmp/vl-jev-probe") / f"request-media{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(response.content)
        context = {key: value for key, value in state.items() if key != "media"}
        return {"type": media["type"], "url": str(path)}, context

    def infer_one(self, media, state_context, question, *, profile=False, full_vocab=False):
        import time

        keys, messages = question_messages(media, state_context, question, self.labels)
        start = time.perf_counter()
        kwargs = {"num_frames": 8} if media and media["type"] == "video" else {}
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        preprocess_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        inputs = inputs.to("cuda")
        self.torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - start) * 1000

        events = {}
        handles = []
        if profile:
            for name, module in (
                ("vision", self.model.model.visual),
                ("language", self.model.model.language_model),
            ):
                events[name] = []

                def pre_hook(_module, _args, stage=name):
                    event = self.torch.cuda.Event(enable_timing=True)
                    event.record()
                    events[stage].append([event, None])

                def post_hook(_module, _args, _output, stage=name):
                    event = self.torch.cuda.Event(enable_timing=True)
                    event.record()
                    events[stage][-1][1] = event

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook))

        start = time.perf_counter()
        try:
            if full_vocab:
                output = self.model(**inputs, logits_to_keep=1, use_cache=False)
                selected_logits = output.logits[0, -1].index_select(
                    0, self.label_token_ids[: len(keys)]
                )
            else:
                output = self.model.model(**inputs, use_cache=False)
                final_hidden = output.last_hidden_state[:, -1, :]
        finally:
            for handle in handles:
                handle.remove()
        self.torch.cuda.synchronize()
        backbone_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        if not full_vocab:
            selected_logits = self.torch.nn.functional.linear(
                final_hidden, self.label_weights[: len(keys)]
            )[0]
        probabilities = selected_logits.float().softmax(dim=-1).cpu().tolist()
        score_ms = (time.perf_counter() - start) * 1000
        diagnostics = {
            "input_tokens": int(inputs["input_ids"].numel()),
            "preprocess_ms": round(preprocess_ms, 2),
            "transfer_ms": round(transfer_ms, 2),
            "backbone_ms": round(backbone_ms, 2),
            "allowed_score_ms": round(score_ms, 2),
            "decoder": "full_vocab" if full_vocab else "allowed_rows",
        }
        if profile:
            diagnostics.update({
                f"{name}_gpu_ms": round(
                    sum(begin.elapsed_time(end) for begin, end in calls), 2
                )
                for name, calls in events.items()
            })
        return make_answer(question, keys, probabilities), diagnostics

    def infer_batch(self, media, state_context, questions):
        import time

        keys_by_question = {}
        conversations = []
        for name, question in questions.items():
            keys, messages = question_messages(media, state_context, question, self.labels)
            keys_by_question[name] = keys
            conversations.append(messages)
        start = time.perf_counter()
        processor_kwargs = {"padding": True}
        if media and media["type"] == "video":
            processor_kwargs["num_frames"] = 8
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )
        preprocess_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        inputs = inputs.to("cuda")
        self.torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - start) * 1000
        if not bool(inputs["attention_mask"][:, 0].all()):
            raise ValueError("Batch padding unexpectedly preceded the state")
        start = time.perf_counter()
        output = self.model.model(**inputs, use_cache=False)
        self.torch.cuda.synchronize()
        backbone_ms = (time.perf_counter() - start) * 1000
        last_positions = inputs["attention_mask"].sum(dim=1).long() - 1
        row_indices = self.torch.arange(len(questions), device="cuda")
        final_hidden = output.last_hidden_state[row_indices, last_positions, :]
        start = time.perf_counter()
        answers = {}
        for index, (name, question) in enumerate(questions.items()):
            keys = keys_by_question[name]
            logits = self.torch.nn.functional.linear(
                final_hidden[index : index + 1], self.label_weights[: len(keys)]
            )[0]
            probabilities = logits.float().softmax(dim=-1).cpu().tolist()
            answers[name] = make_answer(question, keys, probabilities)
        score_ms = (time.perf_counter() - start) * 1000
        return answers, {
            "input_tokens": int(inputs["attention_mask"].sum().item()),
            "preprocess_ms": round(preprocess_ms, 2),
            "transfer_ms": round(transfer_ms, 2),
            "backbone_ms": round(backbone_ms, 2),
            "allowed_score_ms": round(score_ms, 2),
        }

    def _evaluate(self, request, *, profile=False, full_vocab=False):
        import time

        if not isinstance(request, dict) or not all(
            name in request for name in ("state", "model", "questions")
        ):
            raise ValueError("Request must have state, model, and questions")
        if request["model"] != API_MODEL_NAME:
            raise ValueError(f"This experiment serves only {API_MODEL_NAME}")
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty named object")
        if not isinstance(request["state"], (str, dict, list)):
            raise ValueError("state must be text, an object, or an array")
        start = time.perf_counter()
        media, context = self.media_from_state(request["state"])
        media_ms = (time.perf_counter() - start) * 1000
        answers = {}
        details = {}
        start = time.perf_counter()
        input_tokens = 0
        for name, question in questions.items():
            if not isinstance(question, dict):
                raise ValueError(f"Question {name!r} must be an object")
            answer, diagnostic = self.infer_one(
                media, context, question, profile=profile, full_vocab=full_vocab
            )
            answers[name] = answer
            details[name] = diagnostic
            input_tokens += diagnostic["input_tokens"]
        response = {
            "model": API_MODEL_NAME,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        return {
            "response": response,
            "diagnostics": details,
            "media_ms": round(media_ms, 2),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    @modal.method()
    def evaluate(self, request):
        """Return exactly the published Jev response fields."""
        return self._evaluate(request)["response"]

    @modal.method()
    def evaluate_debug(self, request):
        """Return the same response plus experiment-only timings and logits checks."""
        return self._evaluate(request, profile=True)

    @modal.method()
    def evaluate_fast(self, request):
        """Experimental batch path: enum-safe, but it can change the decision."""
        if not isinstance(request, dict) or not all(
            name in request for name in ("state", "model", "questions")
        ) or request["model"] != API_MODEL_NAME:
            raise ValueError("Expected a Jev-shaped request for this model")
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty named object")
        if not isinstance(request["state"], (str, dict, list)):
            raise ValueError("state must be text, an object, or an array")
        media, context = self.media_from_state(request["state"])
        answers, diagnostics = self.infer_batch(media, context, questions)
        return {
            "model": API_MODEL_NAME,
            "answers": answers,
            "usage": {"input_tokens": diagnostics["input_tokens"], "output_tokens": 0},
        }

    @modal.method()
    def check_255_choices(self):
        criteria = {
            f"option_{i:03d}": "A possible category" for i in range(255)
        }
        request = {
            "state": "A neutral test state.",
            "model": API_MODEL_NAME,
            "questions": {
                "selected": {
                    "type": "choice",
                    "instructions": "Select one of the supplied categories.",
                    "criteria": criteria,
                }
            },
        }
        response = self._evaluate(request)["response"]
        answer = response["answers"]["selected"]
        return {
            "option_count": len(criteria),
            "returned_count": len(answer["probabilities"]),
            "selected_is_allowed": answer["choice"] in criteria,
            "keys_match_exactly": set(answer["probabilities"]) == set(criteria),
            "probability_sum": sum(answer["probabilities"].values()),
        }

    @modal.method()
    def compare_decoders(self, request, repeats=4):
        """Profile both exact allowed-row projection and the previous full head."""
        from statistics import median

        if len(request["questions"]) != 1:
            raise ValueError("Decoder comparison requires exactly one question")
        runs = {"allowed_rows": [], "full_vocab": []}
        outputs = {}
        for _ in range(repeats):
            for name, full_vocab in (("allowed_rows", False), ("full_vocab", True)):
                result = self._evaluate(request, profile=True, full_vocab=full_vocab)
                runs[name].append(result)
                outputs[name] = result["response"]["answers"]
        question_id = next(iter(request["questions"]))
        left = outputs["allowed_rows"][question_id]
        right = outputs["full_vocab"][question_id]
        if left["type"] == "choice":
            delta = max(
                abs(left["probabilities"][key] - right["probabilities"][key])
                for key in left["probabilities"]
            )
            same_choice = left["choice"] == right["choice"]
        else:
            delta = abs(left.get("noul", left.get("score")) - right.get("noul", right.get("score")))
            same_choice = True
        return {
            "same_choice": same_choice,
            "max_probability_delta": delta,
            "runs": {
                name: {
                    "elapsed_ms": [run["elapsed_ms"] for run in values],
                    "median_elapsed_ms": round(median(run["elapsed_ms"] for run in values), 2),
                    "median_preprocess_ms": round(median(next(iter(run["diagnostics"].values()))["preprocess_ms"] for run in values), 2),
                    "median_transfer_ms": round(median(next(iter(run["diagnostics"].values()))["transfer_ms"] for run in values), 2),
                    "median_backbone_ms": round(median(next(iter(run["diagnostics"].values()))["backbone_ms"] for run in values), 2),
                    "median_vision_gpu_ms": round(median(next(iter(run["diagnostics"].values()))["vision_gpu_ms"] for run in values), 2),
                    "median_language_gpu_ms": round(median(next(iter(run["diagnostics"].values()))["language_gpu_ms"] for run in values), 2),
                    "median_score_ms": round(median(next(iter(run["diagnostics"].values()))["allowed_score_ms"] for run in values), 2),
                }
                for name, values in runs.items()
            },
        }

    @modal.method()
    def compare_question_batch(self, request, repeats=4):
        from statistics import median
        import time

        if len(request["questions"]) < 2:
            raise ValueError("Batch comparison needs at least two questions")
        media, context = self.media_from_state(request["state"])
        serial_runs = []
        batch_runs = []
        max_delta = 0.0
        same_choices = True
        deltas_by_question = {name: 0.0 for name in request["questions"]}
        for _ in range(repeats):
            start = time.perf_counter()
            serial_answers = {}
            for name, question in request["questions"].items():
                answer, _ = self.infer_one(media, context, question)
                serial_answers[name] = answer
            serial_runs.append(round((time.perf_counter() - start) * 1000, 2))
            start = time.perf_counter()
            batch_answers, batch_diagnostic = self.infer_batch(
                media, context, request["questions"]
            )
            batch_runs.append({
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                **batch_diagnostic,
            })
            for name, serial_answer in serial_answers.items():
                batch_answer = batch_answers[name]
                if serial_answer["type"] == "choice":
                    same_choices &= serial_answer["choice"] == batch_answer["choice"]
                    delta = max(
                        abs(serial_answer["probabilities"][key] - batch_answer["probabilities"][key])
                        for key in serial_answer["probabilities"]
                    )
                elif serial_answer["type"] == "noul":
                    delta = abs(serial_answer["noul"] - batch_answer["noul"])
                else:
                    delta = max(
                        abs(serial_answer["probabilities"][key] - batch_answer["probabilities"][key])
                        for key in serial_answer["probabilities"]
                    )
                max_delta = max(max_delta, delta)
                deltas_by_question[name] = max(deltas_by_question[name], delta)
        return {
            "same_choices": same_choices,
            "max_probability_delta": round(max_delta, 6),
            "max_delta_by_question": {
                name: round(delta, 6) for name, delta in deltas_by_question.items()
            },
            "serial_ms": serial_runs,
            "batch_ms": [run["elapsed_ms"] for run in batch_runs],
            "median_serial_ms": round(median(serial_runs), 2),
            "median_batch_ms": round(median(run["elapsed_ms"] for run in batch_runs), 2),
            "batch_stages_ms": {
                key: round(median(run[key] for run in batch_runs), 2)
                for key in ("preprocess_ms", "transfer_ms", "backbone_ms", "allowed_score_ms")
            },
        }


@app.function(image=image, volumes={"/probe-media": probe_media}, timeout=120)
def create_remote_test_media():
    """A tiny diagram control: only the goal variant crosses the goal line."""
    from pathlib import Path

    import av
    from PIL import Image, ImageDraw

    root = Path("/probe-media")
    root.mkdir(exist_ok=True)
    result = {}
    for goal in (True, False):
        name = "goal" if goal else "miss"
        path = root / f"soccer_{name}.mp4"
        output = av.open(str(path), mode="w")
        stream = output.add_stream("libx264", rate=8)
        stream.width = 224
        stream.height = 224
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            frame = Image.new("RGB", (224, 224), "#218b47")
            draw = ImageDraw.Draw(frame)
            draw.line((160, 20, 160, 204), fill="white", width=3)
            draw.rectangle((160, 54, 215, 150), outline="white", width=5)
            x = 18 + i * 24
            y = 105 if goal else 178
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="white", outline="black", width=2)
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()
        still = root / f"soccer_{name}.png"
        frame.save(still)
        result[name] = {"video": str(path), "image": str(still)}
    probe_media.commit()
    return result


@app.local_entrypoint()
def main():
    # The local process only submits Modal calls and prints JSON. No media or weights
    # are created, downloaded, opened, or processed on the local computer.
    model = DecisionModel()
    evaluate_debug = model.evaluate_debug
    compare_decoders = model.compare_decoders
    compare_question_batch = model.compare_question_batch
    check_255_choices = model.check_255_choices
    media = create_remote_test_media.remote()
    cases = []
    comparison_requests = {}
    batch_requests = {}
    for name, paths in media.items():
        for modality in ("video", "image"):
            questions = {
                "goal_event" if modality == "video" else "ball_position": {
                    "type": "choice",
                    "instructions": "Did the ball cross into the goal?" if modality == "video" else "Where is the ball relative to the goal?",
                    "criteria": {
                        "goal": "The ball enters the goal between the posts.",
                        "miss": "The ball passes outside the goal posts.",
                    },
                },
                "ball_visible": {
                    "type": "noul",
                    "instructions": "Is a ball visible?",
                    "criteria": {"true": "A ball is visible", "false": "No ball is visible"},
                },
                "goal_evidence": {
                    "type": "score",
                    "instructions": "How strong is the visible evidence that a goal occurred?",
                    "criteria": ["No goal evidence", "Ambiguous goal evidence", "Clear goal evidence"],
                },
            }
            request = {
                "state": {"media": {"type": modality, "url": paths[modality]}, "sport": "soccer"},
                "model": API_MODEL_NAME,
                "questions": questions,
            }
            try:
                result = evaluate_debug.remote(request)
                first_question = next(iter(questions))
                answer = result["response"]["answers"][first_question]
                if answer["type"] == "choice" and answer["choice"] not in questions[first_question]["criteria"]:
                    raise AssertionError("Choice escaped the supplied enum")
                cases.append({
                    "case": f"{name}_{modality}",
                    "selected": answer["choice"],
                    "goal_probability": round(answer["probabilities"]["goal"], 4),
                    "elapsed_ms": result["elapsed_ms"],
                    "stages": result["diagnostics"][first_question],
                })
                batch_requests[f"{name}_{modality}"] = request
                if name == "goal":
                    comparison_requests[modality] = {
                        **request,
                        "questions": {first_question: questions[first_question]},
                    }
            except Exception as exc:
                cases.append({"case": f"{name}_{modality}", "error": str(exc)})
    comparisons = {}
    for modality, request in comparison_requests.items():
        comparisons[modality] = compare_decoders.remote(request, repeats=5)
    batches = {}
    for case_name, request in batch_requests.items():
        try:
            batches[case_name] = compare_question_batch.remote(request, repeats=4)
        except Exception as exc:
            batches[case_name] = {"error": str(exc)}
    option_check = check_255_choices.remote()
    print(json.dumps({"cases": cases, "decoder_comparisons": comparisons, "question_batch_comparisons": batches, "option_check": option_check}, indent=2, ensure_ascii=False))
