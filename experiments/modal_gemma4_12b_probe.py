"""Gemma 4 12B Jev-shaped restricted-choice probe; all inference stays on Modal."""

import json
import math
import string
import time

import modal


app = modal.App("vl-jev-gemma4-12b-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.17.0",
        "accelerate==1.13.0",
        "av==16.1.0",
        "pillow==12.1.1",
        "requests==2.32.5",
    )
    .env({"HF_HOME": "/model-cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
codec_image = image.apt_install("ffmpeg").pip_install("torchcodec==0.10.0")
cuda_codec_image = (
    image.apt_install("ffmpeg")
    .pip_install("nvidia-npp-cu12==12.3.3.100")
    .run_commands("pip install 'torchcodec==0.10.0+cu128' --index-url https://download.pytorch.org/whl/cu128")
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/npp/lib:/usr/local/lib/python3.11/site-packages/torch/lib"})
)
cache = modal.Volume.from_name("vl-jev-zero-shot-hf-cache", create_if_missing=True)
probe_media = modal.Volume.from_name("vl-jev-probe-media", create_if_missing=True)
MODEL_NAME = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
API_MODEL_NAME = "vl-jev-gemma4-12b-zero-shot"


def labels_for(tokenizer):
    labels, token_ids = [], []
    candidates = (
        list(string.digits)
        + list(string.ascii_uppercase)
        + list(string.ascii_lowercase)
        + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        + [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
    )
    for label in candidates:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in token_ids and tokenizer.decode(ids) == label:
            labels.append(label)
            token_ids.append(ids[0])
        if len(labels) == 255:
            break
    if len(labels) != 255:
        raise ValueError(f"Only {len(labels)} unique one-token labels")
    return labels, token_ids


def candidates_for(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("choice criteria must contain 1 to 255 named options")
        if any(not isinstance(key, str) or not key for key in criteria):
            raise ValueError("choice names must be nonempty strings")
        return list(criteria), [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in criteria.items()]
    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object or null")
        return ["false", "true"], [
            json.dumps(criteria.get("false", "No"), ensure_ascii=False),
            json.dumps(criteria.get("true", "Yes"), ensure_ascii=False),
        ]
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must contain 2 to 10 ordered levels")
        return [str(i) for i in range(len(criteria))], [json.dumps(x, ensure_ascii=False) for x in criteria]
    raise ValueError(f"Unknown question type: {kind!r}")


def answer_for(question, keys, probabilities):
    if len(keys) != len(probabilities) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid option probabilities")
    if abs(sum(probabilities) - 1) > 1e-4:
        raise ValueError("Option probabilities do not sum to one")
    if question["type"] == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    distribution = dict(zip(keys, probabilities))
    if len(keys) == 1:
        confidence = 1.0
    else:
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        confidence = max(0.0, min(1.0, 1 - entropy / math.log(len(keys))))
    if question["type"] == "choice":
        return {"type": "choice", "choice": keys[max(range(len(keys)), key=lambda i: probabilities[i])],
                "confidence": confidence, "probabilities": distribution}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probabilities)),
            "confidence": confidence, "legend": dict(zip(keys, question["criteria"])),
            "probabilities": distribution}


@app.function(image=image, volumes={"/model-cache": cache}, timeout=300)
def inspect_processor():
    """Inspect Gemma's final-answer prefix and labels without loading weights."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    labels, _ = labels_for(processor.tokenizer)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Choose A or B."}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    cache.commit()
    return {"tail": repr(prompt[-180:]), "label_count": len(labels), "first_labels": labels[:8]}


@app.cls(
    image=codec_image, gpu="L40S", volumes={"/model-cache": cache, "/probe-media": probe_media},
    timeout=1800,
)
class DecisionModel:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        torch.set_grad_enabled(False)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.processor.tokenizer.padding_side = "right"
        self.model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.labels, token_ids = labels_for(self.processor.tokenizer)
        self.label_weights = self.model.lm_head.weight.index_select(
            0, torch.tensor(token_ids, device="cuda", dtype=torch.long)
        ).detach()
        self.cap = self.model.config.get_text_config().final_logit_softcapping
        self._debug_visual_cache = {}
        cache.commit()

    def state_media(self, state):
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
                raise ValueError("Media exceeds 50 MB")
            suffix = ".mp4" if media["type"] == "video" else ".jpg"
            path = Path("/tmp/vl-jev-gemma") / f"media{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(response.content)
        return {"type": media["type"], "url": str(path)}, {k: v for k, v in state.items() if k != "media"}

    def messages_for(self, media, state, question):
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object")
        keys, descriptions = candidates_for(question)
        prompt = (
            "Decide from the supplied state and visual evidence. Give exactly one label.\n"
            f"State: {json.dumps(state, ensure_ascii=False)}\n"
            f"Question: {json.dumps(question.get('instructions', ''), ensure_ascii=False)}\n"
            + "\n".join(f"{label}. {desc}" for label, desc in zip(self.labels, descriptions))
            + "\nAnswer:"
        )
        content = []
        if media:
            content.append({"type": media["type"], "url" if media["type"] == "image" else "video": media["url"]})
        content.append({"type": "text", "text": prompt})
        return keys, [{"role": "user", "content": content}]

    def prepare_visual(self, media, video_soft_tokens):
        """Run media decoding and visual preprocessing once per request."""
        if media["type"] == "image":
            images = self.processor.image_processor.fetch_images([[media["url"]]])
            visual, replacements = self.processor._process_images(
                images, return_tensors="pt"
            )
            return visual, replacements, []
        visual, replacements = self.processor._process_videos(
            [[media["url"]]], num_frames=8, max_soft_tokens=video_soft_tokens,
            do_sample_frames=True, return_metadata=True, return_tensors="pt",
        )
        return visual, [], replacements

    def prepare_visual_gpu(self, media, video_soft_tokens):
        """Decode video on CPU, then transfer sampled frames before Gemma patchification."""
        if media["type"] != "video":
            return self.prepare_visual(media, video_soft_tokens)
        visual, replacements = self.processor._process_videos(
            [[media["url"]]], num_frames=8, max_soft_tokens=video_soft_tokens,
            do_sample_frames=True, return_metadata=True, return_tensors="pt", device="cuda",
        )
        self.torch.cuda.synchronize()
        return visual, [], replacements

    def shared_inputs(self, messages, visual_cache):
        """Reuse the processor's exact visual tensors and token replacements."""
        from transformers.image_processing_utils import BatchFeature

        visual, image_replacements, video_replacements = visual_cache
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        text, _ = self.processor.get_text_with_replacements(
            [rendered], image_replacements, video_replacements, []
        )
        tokenized = self.processor.tokenizer(
            text, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        tokenized["mm_token_type_ids"] = self.torch.as_tensor(
            self.processor.create_mm_token_type_ids(tokenized["input_ids"]),
            dtype=self.torch.long,
        )
        data = {**tokenized, **visual}
        data.pop("video_metadata", None)
        for name in self.processor.unused_input_names:
            data.pop(name, None)
        return BatchFeature(data)

    def infer_one(self, media, state, question, *, video_soft_tokens=70,
                  visual_cache=None, return_logits=False):
        keys, messages = self.messages_for(media, state, question)
        start = time.perf_counter()
        if visual_cache is None:
            kwargs = {"num_frames": 8, "max_soft_tokens": video_soft_tokens} if media and media["type"] == "video" else {}
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt", processor_kwargs=kwargs,
            )
        else:
            inputs = self.shared_inputs(messages, visual_cache)
        preprocess_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        inputs = inputs.to("cuda")
        self.torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - start) * 1000
        events = {}
        handles = []
        for name, module in (("vision", self.model.model.embed_vision),
                             ("language", self.model.model.language_model)):
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
            output = self.model.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        self.torch.cuda.synchronize()
        backbone_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        logits = self.torch.nn.functional.linear(
            output.last_hidden_state[:, -1, :], self.label_weights[:len(keys)]
        )[0]
        if self.cap is not None:
            logits = self.torch.tanh(logits / self.cap) * self.cap
        probabilities = logits.float().softmax(dim=-1).cpu().tolist()
        score_ms = (time.perf_counter() - start) * 1000
        diagnostics = {
            "input_tokens": int(inputs["input_ids"].numel()),
            "visual_input_shapes": {
                key: list(value.shape) for key, value in inputs.items()
                if ("pixel" in key or "video" in key) and hasattr(value, "shape")
            },
            "preprocess_ms": round(preprocess_ms, 2),
            "transfer_ms": round(transfer_ms, 2),
            "backbone_ms": round(backbone_ms, 2),
            "allowed_score_ms": round(score_ms, 2),
            "vision_gpu_ms": round(sum(a.elapsed_time(b) for a, b in events["vision"]), 2),
            "language_gpu_ms": round(sum(a.elapsed_time(b) for a, b in events["language"]), 2),
        }
        answer = answer_for(question, keys, probabilities)
        if return_logits:
            return answer, diagnostics, logits.float().cpu().tolist()
        return answer, diagnostics

    @modal.method()
    def compare_calibration(self, request):
        """Research probe: subtract a text-only option prior, without generating."""
        if request.get("model") != API_MODEL_NAME or len(request.get("questions", {})) != 1:
            raise ValueError("Calibration probe needs this model and one question")
        media, state = self.state_media(request["state"])
        if media is None:
            raise ValueError("Calibration probe needs image or video media")
        question = next(iter(request["questions"].values()))
        keys, _ = candidates_for(question)
        _, media_profile, media_logits = self.infer_one(
            media, state, question, return_logits=True
        )
        _, null_profile, null_logits = self.infer_one(
            None, state, question, return_logits=True
        )
        scores = {}
        for weight in (0.0, 0.5, 1.0):
            adjusted = self.torch.tensor(media_logits) - weight * self.torch.tensor(null_logits)
            probs = adjusted.softmax(dim=-1).tolist()
            scores[str(weight)] = answer_for(question, keys, probs)
        return {"media_logits": dict(zip(keys, media_logits)),
                "null_logits": dict(zip(keys, null_logits)),
                "answers_by_prior_weight": scores,
                "media_profile": media_profile, "null_profile": null_profile}

    @modal.method()
    def compare_balanced_order(self, request):
        """Research probe: each Choice key occupies every label position once."""
        if request.get("model") != API_MODEL_NAME or len(request.get("questions", {})) != 1:
            raise ValueError("Balanced probe needs this model and one question")
        media, state = self.state_media(request["state"])
        question = next(iter(request["questions"].values()))
        if question.get("type") != "choice":
            raise ValueError("Balanced probe currently supports Choice")
        keys, _ = candidates_for(question)
        if len(keys) > 8:
            raise ValueError("Balanced probe is limited to eight options")
        visual_cache = self.prepare_visual(media, 70) if media else None
        scores = {key: 0.0 for key in keys}
        profiles = []
        for shift in range(len(keys)):
            ordered = keys[shift:] + keys[:shift]
            rotated = {**question, "criteria": {key: question["criteria"][key] for key in ordered}}
            _, profile, logits = self.infer_one(
                media, state, rotated, visual_cache=visual_cache, return_logits=True
            )
            for key, logit in zip(ordered, logits):
                scores[key] += logit / len(keys)
            profiles.append(profile)
        probabilities = self.torch.tensor([scores[key] for key in keys]).softmax(dim=-1).tolist()
        return {"answer": answer_for(question, keys, probabilities),
                "mean_logits": scores,
                "backbone_ms": round(sum(p["backbone_ms"] for p in profiles), 2),
                "preprocess_ms": round(sum(p["preprocess_ms"] for p in profiles), 2)}

    def _evaluate(self, request, *, video_soft_tokens=70, shared_visual=True):
        if not isinstance(request, dict) or not all(k in request for k in ("state", "model", "questions")):
            raise ValueError("Request must have state, model, and questions")
        if request["model"] != API_MODEL_NAME:
            raise ValueError(f"Expected model {API_MODEL_NAME}")
        if not isinstance(request["state"], (str, dict, list)):
            raise ValueError("state must be text, object, or array")
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty named object")
        start = time.perf_counter()
        media, context = self.state_media(request["state"])
        media_ms = (time.perf_counter() - start) * 1000
        answers, diagnostics, input_tokens = {}, {}, 0
        start = time.perf_counter()
        visual_cache = None
        shared_visual_ms = 0.0
        if shared_visual and media and len(questions) > 1:
            visual_cache = self.prepare_visual(media, video_soft_tokens)
            shared_visual_ms = (time.perf_counter() - start) * 1000
        for name, question in questions.items():
            answer, profile = self.infer_one(
                media, context, question, video_soft_tokens=video_soft_tokens,
                visual_cache=visual_cache,
            )
            answers[name] = answer
            diagnostics[name] = profile
            input_tokens += profile["input_tokens"]
        response = {"model": API_MODEL_NAME, "answers": answers,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
        return {"response": response, "diagnostics": diagnostics,
                "media_ms": round(media_ms, 2), "shared_visual_ms": round(shared_visual_ms, 2),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)}

    @modal.method()
    def evaluate(self, request):
        return self._evaluate(request)["response"]

    @modal.method()
    def evaluate_debug(self, request):
        return self._evaluate(request)

    @modal.method()
    def evaluate_serial_debug(self, request):
        """Reference path for parity checks; reprocesses media per question."""
        return self._evaluate(request, shared_visual=False)

    @modal.method()
    def evaluate_debug_140(self, request):
        return self._evaluate(request, video_soft_tokens=140)

    @modal.method()
    def evaluate_shared_debug(self, request):
        return self._evaluate(request, shared_visual=True)

    @modal.method()
    def compare_preprocessing(self, request):
        """Require exact input tensors before accepting the shared preprocessing path."""
        media, state = self.state_media(request["state"])
        if not media:
            raise ValueError("Comparison needs image or video state")
        visual_cache = self.prepare_visual(media, 70)
        results = {}
        for name, question in request["questions"].items():
            _, messages = self.messages_for(media, state, question)
            original = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"num_frames": 8, "max_soft_tokens": 70} if media["type"] == "video" else {},
            )
            shared = self.shared_inputs(messages, visual_cache)
            keys_match = set(original) == set(shared)
            results[name] = {
                "keys_match": keys_match,
                "exact_tensors": keys_match and all(
                    self.torch.equal(original[key], shared[key]) for key in original
                ),
                "input_tokens": int(original["input_ids"].numel()),
            }
        return results

    @modal.method()
    def evaluate_batch_debug(self, request):
        """Experimental single backbone pass for multiple independent questions."""
        if not isinstance(request, dict) or request.get("model") != API_MODEL_NAME:
            raise ValueError("Expected a Jev-shaped request for this model")
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty named object")
        media, state = self.state_media(request["state"])
        names, keys_by_name, conversations = [], {}, []
        for name, question in questions.items():
            keys, messages = self.messages_for(media, state, question)
            names.append(name)
            keys_by_name[name] = keys
            conversations.append(messages)
        start = time.perf_counter()
        kwargs = {"padding": True}
        if media and media["type"] == "video":
            kwargs.update({"num_frames": 8, "max_soft_tokens": 70})
        inputs = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt", processor_kwargs=kwargs,
        )
        preprocess_ms = (time.perf_counter() - start) * 1000
        if "attention_mask" not in inputs:
            raise ValueError("Batch processor did not return attention_mask")
        lengths = inputs["attention_mask"].sum(dim=1).long()
        if not bool(inputs["attention_mask"][:, 0].all()):
            raise ValueError("Expected right padding")
        start = time.perf_counter()
        inputs = inputs.to("cuda")
        self.torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        output = self.model.model(**inputs, use_cache=False)
        self.torch.cuda.synchronize()
        backbone_ms = (time.perf_counter() - start) * 1000
        row_indices = self.torch.arange(len(names), device="cuda")
        hidden = output.last_hidden_state[row_indices, lengths.to("cuda") - 1, :]
        start = time.perf_counter()
        answers = {}
        for index, name in enumerate(names):
            question = questions[name]
            keys = keys_by_name[name]
            logits = self.torch.nn.functional.linear(
                hidden[index:index + 1], self.label_weights[:len(keys)]
            )[0]
            if self.cap is not None:
                logits = self.torch.tanh(logits / self.cap) * self.cap
            probabilities = logits.float().softmax(dim=-1).cpu().tolist()
            answers[name] = answer_for(question, keys, probabilities)
        score_ms = (time.perf_counter() - start) * 1000
        return {"response": {"model": API_MODEL_NAME, "answers": answers,
                              "usage": {"input_tokens": int(lengths.sum().item()), "output_tokens": 0}},
                "preprocess_ms": round(preprocess_ms, 2),
                "transfer_ms": round(transfer_ms, 2),
                "backbone_ms": round(backbone_ms, 2),
                "allowed_score_ms": round(score_ms, 2),
                "elapsed_ms": round(preprocess_ms + transfer_ms + backbone_ms + score_ms, 2)}

    @modal.method()
    def evaluate_prefix_batch_debug(self, request, gpu_patches=False, reuse_visual=False):
        """Research path: prefill the common visual prefix once, then batch question suffixes."""
        if request.get("model") != API_MODEL_NAME:
            raise ValueError("Expected this model")
        questions = request.get("questions")
        if not isinstance(questions, dict) or len(questions) < 2:
            raise ValueError("Prefix probe needs at least two named questions")
        media, state = self.state_media(request["state"])
        started = time.perf_counter()
        cache_key = (media["type"], media["url"], gpu_patches) if media else None
        cache_hit = bool(reuse_visual and media and media["url"].startswith("/probe-media/")
                         and cache_key in self._debug_visual_cache)
        if cache_hit:
            visual_cache = self._debug_visual_cache[cache_key]
        else:
            visual_cache = (self.prepare_visual_gpu(media, 70) if gpu_patches
                            else self.prepare_visual(media, 70)) if media else None
            if reuse_visual and media and media["url"].startswith("/probe-media/"):
                self._debug_visual_cache = {cache_key: visual_cache}
        visual_ms = (time.perf_counter() - started) * 1000
        names, keys_by_name, rows = [], {}, []
        started = time.perf_counter()
        for name, question in questions.items():
            keys, messages = self.messages_for(media, state, question)
            row = self.shared_inputs(messages, visual_cache) if visual_cache else self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt",
            )
            names.append(name)
            keys_by_name[name] = keys
            rows.append(row)
        prep_ms = (time.perf_counter() - started) * 1000
        ids = [row["input_ids"][0].tolist() for row in rows]
        lcp = min(len(row) for row in ids)
        for column in range(lcp):
            if any(row[column] != ids[0][column] for row in ids[1:]):
                lcp = column
                break
        if lcp < 2:
            raise ValueError("No reusable prefix")
        prefix = {key: value for key, value in rows[0].items()
                  if key not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in rows[0]:
                prefix[key] = rows[0][key][:, :lcp]
        for row in rows[1:]:
            if "mm_token_type_ids" in row and not self.torch.equal(
                row["mm_token_type_ids"][:, :lcp], prefix["mm_token_type_ids"]
            ):
                raise ValueError("Multimodal token types differ within common prefix")
        suffix_lengths = [len(row) - lcp for row in ids]
        if min(suffix_lengths) < 1:
            raise ValueError("A question has no unique suffix")
        max_suffix = max(suffix_lengths)
        pad_id = self.processor.tokenizer.pad_token_id
        suffix_ids = self.torch.full((len(rows), max_suffix), pad_id, dtype=self.torch.long)
        suffix_types = self.torch.zeros_like(suffix_ids)
        suffix_mask = self.torch.zeros_like(suffix_ids)
        for index, row in enumerate(rows):
            length = suffix_lengths[index]
            suffix_ids[index, :length] = row["input_ids"][0, lcp:]
            suffix_mask[index, :length] = 1
            if "mm_token_type_ids" in row:
                suffix_types[index, :length] = row["mm_token_type_ids"][0, lcp:]
        prefix = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in prefix.items()}
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        prefix_output = self.model.model(**prefix, use_cache=True)
        self.torch.cuda.synchronize()
        prefix_ms = (time.perf_counter() - started) * 1000
        prefix_cache = prefix_output.past_key_values
        if prefix_cache is None:
            raise ValueError("Model did not return a prefix KV cache")
        started = time.perf_counter()
        prefix_cache.batch_repeat_interleave(len(rows))
        attention_mask = self.torch.cat((
            self.torch.ones((len(rows), lcp), dtype=self.torch.long), suffix_mask
        ), dim=1).to("cuda")
        suffix_inputs = {"input_ids": suffix_ids.to("cuda"),
                         "mm_token_type_ids": suffix_types.to("cuda"),
                         "attention_mask": attention_mask}
        self.torch.cuda.synchronize()
        cache_expand_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        suffix_output = self.model.model(
            **suffix_inputs, past_key_values=prefix_cache, use_cache=True
        )
        self.torch.cuda.synchronize()
        suffix_ms = (time.perf_counter() - started) * 1000
        hidden = suffix_output.last_hidden_state[
            self.torch.arange(len(rows), device="cuda"),
            self.torch.tensor(suffix_lengths, device="cuda") - 1,
        ]
        answers = {}
        for index, name in enumerate(names):
            keys = keys_by_name[name]
            logits = self.torch.nn.functional.linear(
                hidden[index:index + 1], self.label_weights[:len(keys)]
            )[0]
            if self.cap is not None:
                logits = self.torch.tanh(logits / self.cap) * self.cap
            answers[name] = answer_for(
                questions[name], keys, logits.float().softmax(dim=-1).cpu().tolist()
            )
        return {"response": {"model": API_MODEL_NAME, "answers": answers,
                              "usage": {"input_tokens": sum(map(len, ids)), "output_tokens": 0}},
                "prefix_tokens": lcp, "suffix_tokens": suffix_lengths,
                "gpu_patches": gpu_patches,
                "visual_cache_hit": cache_hit,
                "visual_ms": round(visual_ms, 2), "question_prep_ms": round(prep_ms, 2),
                "prefix_ms": round(prefix_ms, 2), "cache_expand_ms": round(cache_expand_ms, 2),
                "suffix_ms": round(suffix_ms, 2),
                "elapsed_ms": round(visual_ms + prep_ms + prefix_ms + cache_expand_ms + suffix_ms, 2)}

    @modal.method()
    def debug_next_tokens(self, request):
        """Inspect top next-token logits; never generate answer text."""
        media, state = self.state_media(request["state"])
        question = next(iter(request["questions"].values()))
        keys, messages = self.messages_for(media, state, question)
        kwargs = {"num_frames": 8, "max_soft_tokens": 70} if media and media["type"] == "video" else {}
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt", processor_kwargs=kwargs,
        ).to("cuda")
        output = self.model.model(**inputs, use_cache=False)
        logits = self.model.lm_head(output.last_hidden_state[:, -1, :])[0].float()
        if self.cap is not None:
            logits = self.torch.tanh(logits / self.cap) * self.cap
        values, indices = logits.topk(12)
        return {"candidate_keys": keys,
                "top_tokens": [{"token": self.processor.tokenizer.decode([int(index)]),
                                "logit": round(float(value), 3)}
                               for value, index in zip(values, indices)],
                "allowed_label_logits": [round(float(logits[self.processor.tokenizer.encode(label, add_special_tokens=False)[0]]), 3)
                                         for label in self.labels[:len(keys)]]}

    @modal.method()
    def check_255_choices(self):
        criteria = {f"option_{i:03d}": "A possible category" for i in range(255)}
        request = {"state": "A neutral test state", "model": API_MODEL_NAME,
                   "questions": {"selected": {"type": "choice", "instructions": "Select a category", "criteria": criteria}}}
        answer = self._evaluate(request)["response"]["answers"]["selected"]
        return {"selected_is_allowed": answer["choice"] in criteria,
                "keys_match_exactly": set(answer["probabilities"]) == set(criteria),
                "probability_sum": sum(answer["probabilities"].values())}


@app.function(image=image, volumes={"/probe-media": probe_media}, timeout=120)
def create_temporal_test_media():
    """Two remote clips with the same last frame; only one enters the goal."""
    from pathlib import Path
    import av
    from PIL import Image, ImageDraw

    root = Path("/probe-media")
    root.mkdir(exist_ok=True)
    paths = {}
    for name, positions in {
        "cross_and_return": [20, 50, 80, 120, 150, 180, 130, 100],
        "approach_and_return": [20, 50, 80, 120, 150, 150, 130, 100],
    }.items():
        path = root / f"soccer_temporal_{name}.mp4"
        output = av.open(str(path), mode="w")
        stream = output.add_stream("libx264", rate=8)
        stream.width = stream.height = 224
        stream.pix_fmt = "yuv420p"
        for x in positions:
            frame = Image.new("RGB", (224, 224), "#218b47")
            draw = ImageDraw.Draw(frame)
            draw.line((160, 20, 160, 204), fill="white", width=3)
            draw.rectangle((160, 54, 215, 150), outline="white", width=5)
            draw.ellipse((x - 8, 97, x + 8, 113), fill="white", outline="black", width=2)
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()
        paths[name] = str(path)
    probe_media.commit()
    return paths


@app.function(image=image, volumes={"/probe-media": probe_media}, timeout=1800)
def prepare_public_sports_sample():
    """Download nine labeled research clips directly into a Modal volume."""
    import csv
    import shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    repo = "infactory-ai/soccer-events"
    metadata_path = hf_hub_download(repo, filename="metadata.csv", repo_type="dataset",
                                    cache_dir="/probe-media/public-sports-cache")
    with open(metadata_path, newline="") as source:
        rows = list(csv.DictReader(source))
    selected = [row for category in ("goal", "yellow_card", "red_card")
                for row in [r for r in rows if r["event_category"] == category][:3]]
    cases = []
    for row in selected:
        path = hf_hub_download(repo, filename=f"data/{row['mp4_file']}", repo_type="dataset",
                               cache_dir="/probe-media/public-sports-cache")
        destination = Path("/probe-media/public-sports") / row["mp4_file"]
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        cases.append({"path": str(destination), "expected": row["event_category"],
                      "duration_seconds": float(row["duration_seconds"]), "id": row["asset_id"]})
    probe_media.commit()
    return cases


def _profile_video_processor(path):
    """Run the same eight-frame preprocessing without loading the backbone."""
    import hashlib
    import time
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    started = time.perf_counter()
    visual, replacements = processor._process_videos(
        [[path]], num_frames=8, max_soft_tokens=70,
        do_sample_frames=True, return_metadata=True, return_tensors="pt",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
              for key, value in visual.items() if hasattr(value, "numpy")}
    return {"elapsed_ms": round(elapsed_ms, 2), "tensor_hashes": hashes,
            "replacements": replacements}


@app.function(image=image, volumes={"/model-cache": cache, "/probe-media": probe_media}, timeout=300)
def profile_torchvision_video(path):
    return _profile_video_processor(path)


@app.function(image=codec_image, volumes={"/model-cache": cache, "/probe-media": probe_media}, timeout=300)
def profile_torchcodec_video(path):
    return _profile_video_processor(path)


@app.function(image=codec_image, volumes={"/model-cache": cache, "/probe-media": probe_media}, timeout=300)
def profile_video_stages(path, repeats=3):
    """Attribute long-video CPU time to frame retrieval versus Gemma patch preparation."""
    import time
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    video = processor.video_processor
    stage_times = {"fetch_and_sample": [], "patch_preprocess": [], "remaining": [], "total": []}
    original_fetch = video.fetch_videos
    original_preprocess = video._preprocess
    current = {}

    def timed_fetch(*args, **kwargs):
        start = time.perf_counter()
        result = original_fetch(*args, **kwargs)
        current["fetch_and_sample"] = (time.perf_counter() - start) * 1000
        return result

    def timed_preprocess(*args, **kwargs):
        start = time.perf_counter()
        result = original_preprocess(*args, **kwargs)
        current["patch_preprocess"] = (time.perf_counter() - start) * 1000
        return result

    video.fetch_videos = timed_fetch
    video._preprocess = timed_preprocess
    for _ in range(repeats):
        current = {}
        start = time.perf_counter()
        processor._process_videos(
            [[path]], num_frames=8, max_soft_tokens=70,
            do_sample_frames=True, return_metadata=True, return_tensors="pt",
        )
        total = (time.perf_counter() - start) * 1000
        fetch = current.get("fetch_and_sample", 0.0)
        patches = current.get("patch_preprocess", 0.0)
        for key, value in (("fetch_and_sample", fetch), ("patch_preprocess", patches),
                           ("remaining", total - fetch - patches), ("total", total)):
            stage_times[key].append(round(value, 2))
    return stage_times


@app.function(image=codec_image, volumes={"/probe-media": probe_media}, timeout=300)
def profile_torchcodec_modes(path):
    """Compare indexed frame retrieval knobs; include frame hashes to catch changed inputs."""
    import hashlib
    import time
    import torch
    from torchcodec.decoders import VideoDecoder

    results = {}
    for seek_mode, threads in (("exact", 0), ("exact", 1), ("exact", 4),
                               ("approximate", 0)):
        key = f"{seek_mode}_{threads}"
        trials = []
        for _ in range(3):
            start = time.perf_counter()
            decoder = VideoDecoder(path, seek_mode=seek_mode, num_ffmpeg_threads=threads)
            total_frames = decoder.metadata.num_frames
            indices = torch.arange(0, total_frames, total_frames / 8).int()
            frames = decoder.get_frames_at(indices=indices).data.contiguous()
            trials.append({"ms": round((time.perf_counter() - start) * 1000, 2),
                           "hash": hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
                           "frame_count": len(indices)})
        results[key] = trials
    reference_hash = results["exact_0"][-1]["hash"]
    return {"results": results,
            "all_exact_identical": all(trial["hash"] == reference_hash
                                       for key, trials in results.items() if key.startswith("exact")
                                       for trial in trials),
            "approximate_identical": all(trial["hash"] == reference_hash
                                          for trial in results["approximate_0"])}


@app.function(image=codec_image, gpu="L40S", volumes={"/probe-media": probe_media}, timeout=300)
def profile_cuda_decode(path):
    """Check whether this pinned TorchCodec build and GPU can use NVDEC."""
    import hashlib
    import time
    import torch
    from torchcodec.decoders import VideoDecoder

    results = {}
    for device in ("cpu", "cuda"):
        trials = []
        try:
            for _ in range(2):
                start = time.perf_counter()
                decoder = VideoDecoder(path, device=device, seek_mode="exact", num_ffmpeg_threads=0)
                count = decoder.metadata.num_frames
                indices = torch.arange(0, count, count / 8).int()
                frames = decoder.get_frames_at(indices=indices).data.contiguous()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                trials.append({"ms": round(elapsed_ms, 2), "tensor_device": str(frames.device),
                               "hash": hashlib.sha256(frames.cpu().numpy().tobytes()).hexdigest()})
            results[device] = trials
        except Exception as error:
            results[device] = {"error": str(error)[:500]}
    return {"torch_cuda_version": torch.version.cuda, "results": results}


@app.function(image=cuda_codec_image, gpu="L40S", volumes={"/probe-media": probe_media}, timeout=300)
def profile_cuda_wheel(path):
    """Test the CUDA 12.8 TorchCodec wheel against the exact CPU frame indices."""
    import hashlib
    import time
    import torch
    from torchcodec.decoders import VideoDecoder

    results = {}
    for device in ("cpu", "cuda"):
        try:
            trials = []
            for _ in range(2):
                start = time.perf_counter()
                decoder = VideoDecoder(path, device=device, seek_mode="exact", num_ffmpeg_threads=0)
                count = decoder.metadata.num_frames
                indices = torch.arange(0, count, count / 8).int()
                frames = decoder.get_frames_at(indices=indices).data.contiguous()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                trials.append({"ms": round(elapsed_ms, 2), "tensor_device": str(frames.device),
                               "hash": hashlib.sha256(frames.cpu().numpy().tobytes()).hexdigest()})
            results[device] = trials
        except Exception as error:
            results[device] = {"error": str(error)[:500]}
    return {"torch_cuda_version": torch.version.cuda, "results": results}


@app.function(image=codec_image, gpu="L40S", volumes={"/model-cache": cache, "/probe-media": probe_media}, timeout=300)
def profile_patch_device(path):
    """Compare CPU and GPU Gemma patch preprocessing from the same decoded frames."""
    import hashlib
    import time
    import torch
    from torchcodec.decoders import VideoDecoder
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    decoder = VideoDecoder(path, seek_mode="exact", num_ffmpeg_threads=0)
    count = decoder.metadata.num_frames
    indices = torch.arange(0, count, count / 8).int()
    frames = decoder.get_frames_at(indices=indices).data.contiguous()
    results = {}
    for device in ("cpu", "cuda"):
        trials = []
        try:
            for _ in range(3):
                started = time.perf_counter()
                processed = processor.video_processor.preprocess(
                    videos=[frames], do_sample_frames=False, max_soft_tokens=70,
                    return_metadata=False, return_tensors="pt", device=device,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                hashes = {key: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                          for key, value in processed.items() if hasattr(value, "cpu")}
                trials.append({"ms": round(elapsed_ms, 2), "hashes": hashes})
            results[device] = trials
        except Exception as error:
            results[device] = {"error": str(error)[:500]}
    return {"results": results,
            "identical": isinstance(results.get("cuda"), list)
                         and results["cpu"][-1]["hashes"] == results["cuda"][-1]["hashes"]}


@app.local_entrypoint()
def main(inspect_only: bool = False, temporal_only: bool = False,
         real_sports: bool = False, shared_probe: bool = False,
         batch_probe: bool = False, decoder_probe: bool = False,
         calibration_probe: bool = False, video_decoder_probe: bool = False,
         balanced_probe: bool = False, prefix_probe: bool = False,
         video_stage_probe: bool = False, codec_modes_probe: bool = False,
         prefix_sports_probe: bool = False, cuda_decode_probe: bool = False,
         patch_device_probe: bool = False, cuda_wheel_probe: bool = False,
         gpu_pipeline_probe: bool = False, warm_visual_probe: bool = False,
         h100_probe: bool = False, distinct_probe: bool = False):
    # This process only submits jobs. Weights and media remain on Modal.
    if inspect_only:
        print(json.dumps(inspect_processor.remote(), indent=2))
        return
    if warm_visual_probe or h100_probe or distinct_probe:
        cases = prepare_public_sports_sample.remote()
        model = DecisionModel.with_options(gpu="H100")() if h100_probe or distinct_probe else DecisionModel()
        criteria = {"goal": "A goal is scored: the ball goes into the goal.",
                    "yellow_card": "The referee shows a player a yellow card.",
                    "red_card": "The referee shows a player a red card."}
        for case in ((cases[4],) if h100_probe or distinct_probe else (cases[4], cases[7])):
            questions = {}
            if distinct_probe:
                prompts = (
                    "Did the ball enter the goal?", "Was a yellow card shown?",
                    "Was a red card shown?", "Is the ball visible?",
                    "Is the referee visible?", "Did a player fall?",
                    "Did players celebrate?", "Is this a replay?",
                    "Was there a corner kick?", "Was there a penalty kick?",
                    "Did a goalkeeper make a save?", "Did the ball go out of play?",
                    "Is the scoreboard visible?", "Did the camera cut to the crowd?",
                    "Was there a free kick?", "Is the goal net visible?",
                )
                for index, prompt in enumerate(prompts):
                    questions[f"question_{index:02d}"] = {
                        "type": "choice", "instructions": prompt,
                        "criteria": {"yes": f"The video shows: {prompt}",
                                     "no": f"The video does not show: {prompt}"},
                    }
            else:
                for index in range(16):
                    if index % 3 == 0:
                        question = {"type": "choice", "instructions": "Which main soccer event is shown?",
                                    "criteria": criteria}
                    elif index % 3 == 1:
                        question = {"type": "choice", "instructions": "Which main soccer event is visible?",
                                    "criteria": dict(reversed(list(criteria.items())))}
                    else:
                        question = {"type": "choice", "instructions": "Did a goal happen?",
                                    "criteria": {"yes": "The ball enters the goal.",
                                                 "no": "The ball does not enter the goal."}}
                    questions[f"question_{index:02d}"] = question
            request = {"state": {"media": {"type": "video", "url": case["path"]}},
                       "model": API_MODEL_NAME, "questions": questions}
            first = model.evaluate_prefix_batch_debug.remote(request, reuse_visual=True)
            second = model.evaluate_prefix_batch_debug.remote(request, reuse_visual=True)
            print(json.dumps({"gpu": "H100" if h100_probe or distinct_probe else "L40S",
                              "distinct_questions": distinct_probe,
                              "id": case["id"], "duration_seconds": case["duration_seconds"],
                              "cold_visual": {k: v for k, v in first.items() if k != "response"},
                              "cached_visual": {k: v for k, v in second.items() if k != "response"},
                              "answers_identical": first["response"]["answers"] == second["response"]["answers"]}))
        return
    if gpu_pipeline_probe:
        cases = prepare_public_sports_sample.remote()
        model = DecisionModel()
        for case in (cases[0], cases[4], cases[7]):
            criteria = {"goal": "A goal is scored: the ball goes into the goal.",
                        "yellow_card": "The referee shows a player a yellow card.",
                        "red_card": "The referee shows a player a red card."}
            request = {"state": {"media": {"type": "video", "url": case["path"]}},
                       "model": API_MODEL_NAME,
                       "questions": {
                           "main_event": {"type": "choice", "instructions": "Which main soccer event is shown?",
                                          "criteria": criteria},
                           "reverse_order": {"type": "choice", "instructions": "Which main soccer event is shown?",
                                             "criteria": dict(reversed(list(criteria.items())))},
                           "goal_yes_no": {"type": "choice", "instructions": "Did a goal happen?",
                                           "criteria": {"yes": "The ball enters the goal.",
                                                        "no": "The ball does not enter the goal."}},
                       }}
            cpu = model.evaluate_prefix_batch_debug.remote(request)
            gpu = model.evaluate_prefix_batch_debug.remote(request, gpu_patches=True)
            a, b = cpu["response"]["answers"], gpu["response"]["answers"]
            print(json.dumps({"id": case["id"], "expected": case["expected"],
                              "cpu_ms": cpu["elapsed_ms"], "gpu_ms": gpu["elapsed_ms"],
                              "cpu_visual_ms": cpu["visual_ms"], "gpu_visual_ms": gpu["visual_ms"],
                              "cpu_choices": {name: value["choice"] for name, value in a.items()},
                              "gpu_choices": {name: value["choice"] for name, value in b.items()},
                              "largest_probability_gap": max(
                                  abs(a[name]["probabilities"][key] - b[name]["probabilities"][key])
                                  for name in request["questions"] for key in a[name]["probabilities"]
                              )}))
        return
    if cuda_wheel_probe:
        cases = prepare_public_sports_sample.remote()
        for case in (cases[0], cases[7]):
            print(json.dumps({"id": case["id"], "duration_seconds": case["duration_seconds"],
                              **profile_cuda_wheel.remote(case["path"])}))
        return
    if patch_device_probe:
        cases = prepare_public_sports_sample.remote()
        for case in (cases[0], cases[4]):
            print(json.dumps({"id": case["id"], "duration_seconds": case["duration_seconds"],
                              **profile_patch_device.remote(case["path"])}))
        return
    if cuda_decode_probe:
        cases = prepare_public_sports_sample.remote()
        for case in (cases[0], cases[7]):
            print(json.dumps({"id": case["id"], "duration_seconds": case["duration_seconds"],
                              **profile_cuda_decode.remote(case["path"])}))
        return
    if prefix_sports_probe:
        cases = prepare_public_sports_sample.remote()
        model = DecisionModel()
        for case in (cases[0], cases[4], cases[7]):
            criteria = {"goal": "A goal is scored: the ball goes into the goal.",
                        "yellow_card": "The referee shows a player a yellow card.",
                        "red_card": "The referee shows a player a red card."}
            questions = {
                "main_event": {"type": "choice", "instructions": "Which main soccer event is shown in this clip?",
                               "criteria": criteria},
                "card_or_goal": {"type": "choice", "instructions": "Identify the event visible in this video.",
                                 "criteria": dict(reversed(list(criteria.items())))},
                "goal_yes_no": {"type": "choice", "instructions": "Did a goal happen?",
                                "criteria": {"yes": "The ball enters the goal.",
                                             "no": "The ball does not enter the goal."}},
            }
            request = {"state": {"media": {"type": "video", "url": case["path"]}},
                       "model": API_MODEL_NAME, "questions": questions}
            reference = model.evaluate_shared_debug.remote(request)
            candidate = model.evaluate_prefix_batch_debug.remote(request)
            original, proposed = reference["response"]["answers"], candidate["response"]["answers"]
            print(json.dumps({"id": case["id"], "expected": case["expected"],
                              "reference_ms": reference["elapsed_ms"],
                              "candidate": {k: value for k, value in candidate.items() if k != "response"},
                              "reference_choices": {name: value["choice"] for name, value in original.items()},
                              "candidate_choices": {name: value["choice"] for name, value in proposed.items()},
                              "largest_probability_gap": max(
                                  abs(original[name]["probabilities"][key] - proposed[name]["probabilities"][key])
                                  for name in questions for key in original[name]["probabilities"]
                              )}))
        return
    if codec_modes_probe:
        cases = prepare_public_sports_sample.remote()
        for case in (cases[0], cases[4], cases[7]):
            print(json.dumps({"id": case["id"], "duration_seconds": case["duration_seconds"],
                              **profile_torchcodec_modes.remote(case["path"])}))
        return
    if video_stage_probe:
        cases = prepare_public_sports_sample.remote()
        for case in (cases[0], cases[4], cases[7]):
            print(json.dumps({"id": case["id"], "duration_seconds": case["duration_seconds"],
                              "stages": profile_video_stages.remote(case["path"])}))
        return
    if prefix_probe:
        model = DecisionModel()
        for modality, suffix in (("video", "mp4"), ("image", "png")):
            questions = {}
            for index in range(16):
                if index % 3 == 0:
                    question = {"type": "choice", "instructions": "Did the ball enter the goal?",
                                "criteria": {"goal": "The ball enters the goal.", "miss": "The ball misses the goal."}}
                elif index % 3 == 1:
                    question = {"type": "noul", "instructions": "Is a ball visible?",
                                "criteria": {"true": "A ball is visible", "false": "No ball is visible"}}
                else:
                    question = {"type": "score", "instructions": "How strong is the visible goal evidence?",
                                "criteria": ["No evidence", "Some evidence", "Clear evidence"]}
                questions[f"question_{index:02d}"] = question
            request = {"state": {"media": {"type": modality,
                                             "url": f"/probe-media/soccer_goal.{suffix}"}},
                       "model": API_MODEL_NAME, "questions": questions}
            reference = model.evaluate_shared_debug.remote(request)
            candidate = model.evaluate_prefix_batch_debug.remote(request)
            original = reference["response"]["answers"]
            proposed = candidate["response"]["answers"]
            largest_probability_gap = max(
                abs(original[name]["probabilities"][key] - proposed[name]["probabilities"][key])
                for name in questions if "probabilities" in original[name]
                for key in original[name]["probabilities"]
            )
            print(json.dumps({"modality": modality,
                              "choices_match": all(original[name].get("choice") == proposed[name].get("choice")
                                                   for name in questions if questions[name]["type"] == "choice"),
                              "answers_identical": original == proposed,
                              "largest_probability_gap": largest_probability_gap,
                              "reference_ms": reference["elapsed_ms"],
                              "candidate": {k: value for k, value in candidate.items() if k != "response"}},
                             indent=2))
        return
    if video_decoder_probe:
        cases = prepare_public_sports_sample.remote()
        for case in (cases[0], cases[-1]):
            old = profile_torchvision_video.remote(case["path"])
            new = profile_torchcodec_video.remote(case["path"])
            print(json.dumps({"id": case["id"], "duration_seconds": case["duration_seconds"],
                              "torchvision_ms": old["elapsed_ms"],
                              "torchcodec_ms": new["elapsed_ms"],
                              "tensor_hashes_identical": old["tensor_hashes"] == new["tensor_hashes"],
                              "replacements_identical": old["replacements"] == new["replacements"]}))
        return
    if balanced_probe:
        cases = prepare_public_sports_sample.remote()
        model = DecisionModel()
        for case in cases:
            request = {"state": {"media": {"type": "video", "url": case["path"]}},
                       "model": API_MODEL_NAME,
                       "questions": {"main_event": {
                           "type": "choice", "instructions": "Which main soccer event is shown in this clip?",
                           "criteria": {
                               "goal": "A goal is scored: the ball goes into the goal.",
                               "yellow_card": "The referee shows a player a yellow card.",
                               "red_card": "The referee shows a player a red card.",
                           },
                       }}}
            result = model.compare_balanced_order.remote(request)
            print(json.dumps({"id": case["id"], "expected": case["expected"],
                              "duration_seconds": case["duration_seconds"], **result}))
        return
    if calibration_probe:
        sports_cases = prepare_public_sports_sample.remote()
        model = DecisionModel()
        for case in sports_cases:
            question = {"type": "choice", "instructions": "Which main soccer event is shown in this clip?",
                        "criteria": {
                            "goal": "A goal is scored: the ball goes into the goal.",
                            "yellow_card": "The referee shows a player a yellow card.",
                            "red_card": "The referee shows a player a red card.",
                        }}
            request = {"state": {"media": {"type": "video", "url": case["path"]}},
                       "model": API_MODEL_NAME, "questions": {"main_event": question}}
            result = model.compare_calibration.remote(request)
            print(json.dumps({"id": case["id"], "expected": case["expected"],
                              "duration_seconds": case["duration_seconds"],
                              **result}))
        return
    if decoder_probe:
        model = DecisionModel()
        for case, modality, path in (
            ("synthetic_goal", "image", "/probe-media/soccer_goal.png"),
            ("broadcast_goal", "video", "/probe-media/public-sports/0293e19b-7281-4f97-b2e9-57b3cee260b2.mp4"),
            ("broadcast_yellow", "video", "/probe-media/public-sports/04417a9b-86fc-4b3b-b386-06c7d00d112e.mp4"),
        ):
            request = {"state": {"media": {"type": modality, "url": path}},
                       "model": API_MODEL_NAME,
                       "questions": {"main_event": {
                           "type": "choice", "instructions": "Which main soccer event is shown?",
                           "criteria": {"goal": "A goal is scored.",
                                        "yellow_card": "A yellow card is shown.",
                                        "red_card": "A red card is shown."},
                       }}}
            print(json.dumps({"case": case, **model.debug_next_tokens.remote(request)}, indent=2))
        return
    if batch_probe:
        model = DecisionModel()
        for modality, suffix in (("video", "mp4"), ("image", "png")):
            request = {"state": {"media": {"type": modality,
                                             "url": f"/probe-media/soccer_goal.{suffix}"}},
                       "model": API_MODEL_NAME,
                       "questions": {
                           "goal": {"type": "choice", "instructions": "Did the ball enter the goal?",
                                    "criteria": {"yes": "The ball enters the goal.", "no": "The ball misses the goal."}},
                           "ball": {"type": "noul", "instructions": "Is the ball visible?"},
                           "evidence": {"type": "score", "instructions": "How strong is the goal evidence?",
                                        "criteria": ["None", "Some", "Clear"]},
                       }}
            shared = model.evaluate_shared_debug.remote(request)
            batch = model.evaluate_batch_debug.remote(request)
            answers = shared["response"]["answers"]
            batched_answers = batch["response"]["answers"]
            choice_matches = all(
                answers[name].get("choice") == batched_answers[name].get("choice")
                for name in answers if answers[name]["type"] == "choice"
            )
            print(json.dumps({"modality": modality, "choice_matches": choice_matches,
                              "answers_identical": answers == batched_answers,
                              "shared_ms": shared["elapsed_ms"], "batch_ms": batch["elapsed_ms"],
                              "batch_stages_ms": {k: batch[k] for k in
                                                  ("preprocess_ms", "transfer_ms", "backbone_ms", "allowed_score_ms")},
                              "shared_answers": answers, "batch_answers": batched_answers}, indent=2))
        return
    if shared_probe:
        model = DecisionModel()
        for modality, suffix in (("video", "mp4"), ("image", "png")):
            questions = {}
            for i in range(16):
                if i % 3 == 0:
                    question = {"type": "choice", "instructions": "Did the ball enter the goal?",
                                "criteria": {"goal": "The ball enters the goal.", "miss": "The ball misses the goal."}}
                elif i % 3 == 1:
                    question = {"type": "noul", "instructions": "Is a ball visible?",
                                "criteria": {"true": "A ball is visible", "false": "No ball is visible"}}
                else:
                    question = {"type": "score", "instructions": "How strong is the visible goal evidence?",
                                "criteria": ["No evidence", "Some evidence", "Clear evidence"]}
                questions[f"question_{i:02d}"] = question
            request = {"state": {"media": {"type": modality,
                                             "url": f"/probe-media/soccer_goal.{suffix}"}},
                       "model": API_MODEL_NAME, "questions": questions}
            checks = model.compare_preprocessing.remote(request)
            serial = model.evaluate_serial_debug.remote(request)
            shared = model.evaluate_shared_debug.remote(request)
            serial_answers = serial["response"]["answers"]
            shared_answers = shared["response"]["answers"]
            print(json.dumps({
                "modality": modality,
                "question_count": len(questions),
                "all_inputs_identical": all(x["keys_match"] and x["exact_tensors"] for x in checks.values()),
                "all_answers_identical": serial_answers == shared_answers,
                "serial_elapsed_ms": serial["elapsed_ms"],
                "shared_elapsed_ms": shared["elapsed_ms"],
                "shared_visual_ms": shared["shared_visual_ms"],
                "serial_preprocess_ms": round(sum(x["preprocess_ms"] for x in serial["diagnostics"].values()), 2),
                "shared_question_preprocess_ms": round(sum(x["preprocess_ms"] for x in shared["diagnostics"].values()), 2),
                "response_bytes": len(json.dumps(shared["response"]).encode()),
            }, indent=2))
        return
    if real_sports:
        sports_cases = prepare_public_sports_sample.remote()
        model = DecisionModel()
        for case in sports_cases:
            request = {
                "state": {"media": {"type": "video", "url": case["path"]}},
                "model": API_MODEL_NAME,
                "questions": {"main_event": {
                    "type": "choice", "instructions": "Which main soccer event is shown in this clip?",
                    "criteria": {
                        "goal": "A goal is scored: the ball goes into the goal.",
                        "yellow_card": "The referee shows a player a yellow card.",
                        "red_card": "The referee shows a player a red card.",
                    },
                }},
            }
            result = model.evaluate_debug.remote(request)
            answer = result["response"]["answers"]["main_event"]
            print(json.dumps({"id": case["id"], "expected": case["expected"],
                              "duration_seconds": case["duration_seconds"],
                              "selected": answer["choice"], "probabilities": answer["probabilities"],
                              "timings": result["diagnostics"]["main_event"]}))
        return
    temporal_cases = create_temporal_test_media.remote()
    model = DecisionModel()
    if not temporal_only:
      for case in ("goal", "miss"):
        for modality, suffix in (("video", "mp4"), ("image", "png")):
            request = {
                "state": {"media": {"type": modality, "url": f"/probe-media/soccer_{case}.{suffix}"}},
                "model": API_MODEL_NAME,
                "questions": {
                    "goal_event": {"type": "choice", "instructions": "Did the ball enter the goal?",
                                   "criteria": {"goal": "The ball enters the goal between the posts.",
                                                "miss": "The ball passes outside the goal posts."}},
                    "ball_visible": {"type": "noul", "instructions": "Is a ball visible?",
                                     "criteria": {"true": "A ball is visible", "false": "No ball is visible"}},
                },
            }
            print(json.dumps({"case": case, "modality": modality,
                              **model.evaluate_debug.remote(request)}, indent=2))
      print(json.dumps({"max_choice": model.check_255_choices.remote()}, indent=2))
    for case, path in temporal_cases.items():
        request = {
            "state": {"media": {"type": "video", "url": path}},
            "model": API_MODEL_NAME,
            "questions": {"ever_entered_goal": {
                "type": "choice",
                "instructions": "At any point in this clip, did the ball cross the goal line and enter the goal?",
                "criteria": {"yes": "The ball crosses the line into the goal at some point, even if it returns.",
                             "no": "The ball never crosses the line into the goal."},
            }},
        }
        print(json.dumps({"temporal_case": case, "visual_tokens_per_frame": 70,
                          **model.evaluate_debug.remote(request)}, indent=2))
        print(json.dumps({"temporal_case": case, "visual_tokens_per_frame": 140,
                          **model.evaluate_debug_140.remote(request)}, indent=2))
        reversed_request = {
            **request,
            "questions": {"ever_entered_goal": {
                **request["questions"]["ever_entered_goal"],
                "criteria": dict(reversed(list(request["questions"]["ever_entered_goal"]["criteria"].items()))),
            }},
        }
        print(json.dumps({"temporal_case": case, "visual_tokens_per_frame": 140,
                          "option_order": "reversed", **model.evaluate_debug_140.remote(reversed_request)}, indent=2))
