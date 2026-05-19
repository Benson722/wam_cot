# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase A — VLM-assisted offline STAGE annotation (Latent-CoT, VLM in the
data-generation loop).

For each episode: uniformly sample K frames from the cam_high video, send
them (multimodal) to the local Qwen3.5-27B served by serve_qwen.py
(OpenAI-compatible :8000/v1; offline, no internet), and ask for the ORDERED
manipulation STAGES with the frame index where each starts. Write
``<dataset>/meta/stages.jsonl`` (parallel to keyframes.jsonl) — this is a
RICHER, SEMANTIC supervision signal than the gripper-switch keyframe
distance, used by the training stage-classification head (Phase B).

Reuses cot_planner's vLLM endpoint constants + image encoder + JSON parser
(no new transport/credential code). Runs where serve_qwen is reachable
(H200 GPU0). ``--recursive`` annotates every task dir under a parent
(curated *_stable symlink set) for multi-task training.

Run (env with serve_qwen reachable, repo root):
  python evaluation/robotwin/qwen_stage_annotate.py \
      --dataset /inspire/.../lerobot_robotwin_eef_aug_500_stable \
      --recursive --frames 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.robocasa.cot_planner import (  # noqa: E402
    HARDCODED_VLLM_BASE_URL,
    HARDCODED_VLLM_MODEL,
    CoTPlanner,
    _cred,
    _img_to_data_url,
)

_SYS = """You are a JSON API. You segment a single-arm/dual-arm robot \
manipulation episode into ORDERED semantic STAGES from sampled video \
frames. A stage is a coherent sub-phase (e.g. "approach object", "grasp", \
"lift / move", "place", "retract").

ABSOLUTE OUTPUT RULES (a parser reads your reply with json.loads):
- Output ONE JSON object and NOTHING else. NO analysis. NO explanation. \
NO markdown. NO per-frame description. NO "Stage 1:" prose. NO "Let's \
think". NO sentences. The ENTIRE reply is the JSON.
- Your VERY FIRST output character MUST be '{' and the VERY LAST MUST \
be '}'.
- Schema EXACTLY: {"stages":[{"name":"<<=4 words>","start_frame":<int>}]}
- 2-6 stages. start_frame strictly increasing, in [0, length). The \
first stage MUST have start_frame 0.
Decide silently in your head; emit only the JSON. Replies containing any \
non-JSON text are rejected."""

# One in-context demonstration: shows the assistant turn is PURE JSON with
# zero reasoning, regardless of content. A stubborn CoT model imitates the
# demonstrated brevity far more than it obeys instructions.
_FEWSHOT_USER = ("TASK: pick up the red cup with the right arm\n"
                 "EPISODE LENGTH (frames): 120\n"
                 "The 4 images are frames at indices [0, 40, 80, 119] "
                 "(chronological).")
_FEWSHOT_ASSISTANT = ('{"stages": [{"name": "approach cup", '
                      '"start_frame": 0}, {"name": "grasp", '
                      '"start_frame": 38}, {"name": "lift", '
                      '"start_frame": 70}, {"name": "retract", '
                      '"start_frame": 100}]}')


def _episodes(meta_dir: Path, episodes_file: str):
    fp = meta_dir / episodes_file
    if not fp.exists():
        raise FileNotFoundError(fp)
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _video_path(dataset: Path, ei: int, cam: str, chunks_size: int = 1000):
    chunk = ei // chunks_size
    return (dataset / "videos" / f"chunk-{chunk:03d}" / cam
            / f"episode_{ei:06d}.mp4")


def _read_frames(mp4: Path, idxs):
    """Decode given frame indices. Try imageio(ffmpeg) then pyav."""
    idxs = sorted(set(int(i) for i in idxs))
    try:
        import imageio.v2 as iio
        rdr = iio.get_reader(str(mp4), "ffmpeg")
        out = []
        for i in idxs:
            try:
                out.append(np.asarray(rdr.get_data(i)))
            except Exception:  # noqa: BLE001
                out.append(np.asarray(rdr.get_data(0)))
        rdr.close()
        return out
    except Exception:  # noqa: BLE001
        pass
    try:
        import imageio.v3 as iio3
        frames = iio3.imread(str(mp4), plugin="pyav")  # [T,H,W,C]
        return [np.asarray(frames[min(i, len(frames) - 1)]) for i in idxs]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"cannot decode {mp4} ({e}); install imageio-ffmpeg or av "
            "(av1 support).")


import re as _re

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def _extract_text(resp):
    """Pull the answer out of an OpenAI-style reply, tolerating Qwen3
    thinking models: prefer `content`; if empty fall back to
    `reasoning_content`; strip any <think>...</think> block (and a
    dangling unmatched <think> with no answer after it). Returns
    (text, finish_reason)."""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    fr = ch.get("finish_reason")
    txt = (msg.get("content") or "").strip()
    if not txt:
        txt = (msg.get("reasoning_content") or "").strip()
    txt = _THINK_RE.sub("", txt)
    # unmatched leading <think> with the answer never produced
    if "<think>" in txt and "</think>" not in txt:
        txt = txt.split("<think>", 1)[0].strip()
    return txt.strip(), fr


def _balanced(s, st):
    """Return the JSON object starting at s[st]=='{', closing the brackets
    if the reply was truncated mid-object (string/array unfinished)."""
    stack, instr, esc, out = [], False, False, []
    for ch in s[st:]:
        out.append(ch)
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return "".join(out)
    frag = "".join(out)
    if instr:
        frag += '"'
    while stack:
        frag += "}" if stack.pop() == "{" else "]"
    return frag


def _extract_json(txt):
    """Recover the stage object from a reply that may contain prose, ```json
    fences, or be TRUNCATED. Tries, in order: strict parse, last balanced
    {...} containing "stages" (brace/bracket aware, truncation-repaired),
    then a regex scrape of name/start_frame pairs. Raises only if even one
    start_frame can't be found."""
    s = (txt or "").strip()
    fence = _re.search(r"```(?:json)?\s*(.*?)```", s, _re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        o = CoTPlanner._parse_json(s)
        if isinstance(o, dict) and "stages" in o:
            return o
    except Exception:  # noqa: BLE001
        pass
    starts = [m.start() for m in _re.finditer(r"\{", s)]
    sp = s.rfind('"stages"')
    order = []
    if sp != -1:
        pre = [i for i in starts if i <= sp]
        if pre:
            order.append(pre[-1])
    order += list(reversed(starts))
    seen = set()
    for st in order:
        if st in seen:
            continue
        seen.add(st)
        frag = _balanced(s, st)
        for cand in (frag, _re.sub(r",\s*([}\]])", r"\1", frag)):
            try:
                o = json.loads(cand)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(o, dict) and "stages" in o:
                return o
    pairs = _re.findall(
        r'"name"\s*:\s*"([^"]*)"\s*,\s*"start_frame"\s*:\s*(\d+)', s)
    if pairs:
        return {"stages": [{"name": n, "start_frame": int(f)}
                           for n, f in pairs]}
    nums = _re.findall(r'"start_frame"\s*:\s*(\d+)', s)
    if nums:
        return {"stages": [{"name": "stage", "start_frame": int(n)}
                           for n in nums]}
    raise ValueError("no JSON stages object found in reply")


_PREFIX = '{"stages": ['  # assistant-prefill seed (--prefill)


def _vlm_stage_call(base_url, model, api_key, task, length, idxs, frames,
                    timeout=600, retries=4, max_tokens=4096, no_think=True,
                    return_raw=False, prefill=False):
    # serve_qwen is a custom server that (probe-confirmed) IGNORES
    # enable_thinking/chat_template_kwargs/`/no_think`/response_format and
    # writes a long per-frame essay, never reaching JSON. We don't rely on
    # the server: (1) a hard prompt that forces the JSON to be the LAST
    # thing in the reply, (2) optional assistant-prefill that literally
    # starts the model's turn with `{"stages": [`, (3) a big budget, (4) a
    # truncation-tolerant extractor that grabs the trailing {...}. The
    # extras are still sent (harmless) for servers that DO honor them.
    hint = " /no_think" if no_think else ""
    content = [{"type": "text", "text":
                f"TASK: {task}\nEPISODE LENGTH (frames): {length}\n"
                f"The {len(frames)} images are frames at indices "
                f"{list(idxs)} (chronological).\n"
                "You may reason briefly, but your reply MUST END WITH the "
                "JSON object and the LAST character MUST be '}'. Put "
                "nothing after the closing brace. Schema: "
                '{"stages":[{"name":"<<=4 words>","start_frame":<int>}]} '
                f"(2-6 stages, start_frame increasing, first=0).{hint}"}]
    for fr_img in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": _img_to_data_url(fr_img)}})
    messages = [{"role": "system", "content": _SYS},
                {"role": "user", "content": _FEWSHOT_USER},
                {"role": "assistant", "content": _FEWSHOT_ASSISTANT},
                {"role": "user", "content": content}]
    if prefill:
        # put words in the model's mouth so it continues a JSON array
        # instead of narrating (works iff the server doesn't re-open a
        # fresh assistant turn; harmless otherwise — extractor still runs).
        messages.append({"role": "assistant", "content": _PREFIX})
    # temperature 0 -> serve_qwen sets do_sample=False -> greedy: faster,
    # deterministic, and best for strict-JSON extraction.
    base_payload = {"model": model, "temperature": 0.0,
                    "max_tokens": int(max_tokens), "stream": False,
                    "messages": messages}
    extras = {"response_format": {"type": "json_object"}}
    if no_think:
        extras["enable_thinking"] = False
        extras["chat_template_kwargs"] = {"enable_thinking": False}
    base = base_url.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1")
                  else "/v1/chat/completions")
    use_extras = True
    last = None
    for a in range(retries):
        payload = dict(base_payload)
        if use_extras:
            payload.update(extras)
        data = json.dumps(payload).encode()
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            txt, fin = _extract_text(resp)
            if not txt:
                raise ValueError(
                    f"empty content (finish_reason={fin}); raise "
                    f"--max-tokens (now {max_tokens})")
            # if prefill and the server returned only the continuation
            # (no leading brace / 'stages'), glue the seed back on.
            if prefill and '"stages"' not in txt[:40]:
                txt = _PREFIX + txt
            return (txt, resp) if return_raw else txt
        except urllib.error.HTTPError as e:  # noqa: PERF203
            # server rejected an unknown field (response_format / thinking)
            # -> drop the extras and retry plainly.
            if use_extras and e.code in (400, 404, 422, 500):
                use_extras = False
                last = e
                continue
            last = e
            time.sleep(min(2 ** a, 8))
        except (urllib.error.URLError, KeyError, TimeoutError, ValueError,
                json.JSONDecodeError) as e:
            last = e
            time.sleep(min(2 ** a, 8))
    raise RuntimeError(f"VLM call failed after {retries}: {last}")


def _norm_stages(raw, length):
    obj = _extract_json(raw)
    st = obj.get("stages") or []
    out = []
    for s in st:
        if isinstance(s, dict):
            nm = str(s.get("name", "stage")).strip()[:40]
            sf = int(round(float(s.get("start_frame", 0))))
        else:
            nm, sf = str(s)[:40], 0
        out.append({"name": nm, "start_frame": max(0, min(sf, length - 1))})
    out.sort(key=lambda x: x["start_frame"])
    # dedup / enforce strictly increasing, first stage at 0
    fixed, prev = [], -1
    for s in out:
        if s["start_frame"] <= prev:
            continue
        fixed.append(s)
        prev = s["start_frame"]
    if not fixed or fixed[0]["start_frame"] != 0:
        fixed = [{"name": "start", "start_frame": 0}] + [
            s for s in fixed if s["start_frame"] > 0]
    return fixed[:6]


def probe(dataset: Path, cam: str, frames_k: int, episodes_file: str,
          base_url, model, api_key, max_tokens, no_think, timeout=600,
          prefill=False):
    """One-shot diagnostic: annotate ONLY episode 0 and DUMP the raw server
    reply + the parsed stages. Use this FIRST to confirm serve_qwen is
    answering in the expected JSON before running the full (recursive)
    pass. Writes nothing."""
    meta = dataset / "meta"
    ep = _episodes(meta, episodes_file)[0]
    ei = int(ep["episode_index"])
    L = int(ep.get("length", 0))
    task = (ep.get("tasks") or [""])[0]
    mp4 = _video_path(dataset, ei, cam)
    print(f"[probe] {dataset.name} ep{ei} L={L} task={task!r}\n"
          f"[probe] video={mp4} exists={mp4.exists()}")
    idxs = np.linspace(0, L - 1, num=min(frames_k, L), dtype=int).tolist()
    t0 = time.time()
    frs = _read_frames(mp4, idxs)
    print(f"[probe] decoded {len(frs)} frames in {time.time()-t0:.1f}s; "
          f"calling VLM (max_tokens={max_tokens}, timeout={timeout}s) "
          "-- this server is slow (~minutes), watch nvidia-smi power/util "
          "to confirm it's alive...", flush=True)
    tc = time.time()
    txt, resp = _vlm_stage_call(base_url, model, api_key, task, L, idxs,
                                frs, max_tokens=max_tokens,
                                no_think=no_think, timeout=timeout,
                                return_raw=True, prefill=prefill)
    print(f"[probe] VLM replied in {time.time()-tc:.1f}s", flush=True)
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    fin = (resp.get("choices") or [{}])[0].get("finish_reason")
    usage = resp.get("usage") or {}
    ctok = int(usage.get("completion_tokens", 0))
    truncated = ctok >= max_tokens
    print("[probe] --- raw server message keys:", list(msg.keys()))
    print(f"[probe] --- finish_reason: {fin}  (UNRELIABLE on this server)")
    print(f"[probe] --- usage: {usage}")
    print(f"[probe] --- TRUNCATED={truncated}  (completion_tokens {ctok} "
          f"{'==' if truncated else '<'} max_tokens {max_tokens}); "
          + ("STILL essaying -> raise --max-tokens" if truncated else
             "model stopped on its own -> if no JSON it just won't comply"))
    print(f"[probe] --- extracted text: {len(txt)} chars")
    print("[probe] --- HEAD (first 500):\n" + txt[:500])
    print("[probe] --- TAIL (last 500) -- the JSON should be HERE:\n"
          + txt[-500:])
    stages = _norm_stages(txt, L)
    print(f"[probe] --- PARSED {len(stages)} stages:")
    for s in stages:
        print(f"          start_frame={s['start_frame']:>4d}  {s['name']}")
    print("[probe] OK -- serve_qwen is answering; safe to run the full "
          "pass (drop --probe).")


def annotate(dataset: Path, cam: str, frames_k: int, episodes_file: str,
             out_name: str, base_url, model, api_key, max_tokens=4096,
             no_think=True, limit=0, debug=False, timeout=600,
             prefill=False, resume=False):
    meta = dataset / "meta"
    eps = _episodes(meta, episodes_file)
    if limit > 0:
        eps = eps[:limit]
    out_fp = meta / out_name
    ok = skip = nstage = 0
    dumped = False
    # --resume: keep episodes already in stages.jsonl, append the rest.
    # Lets a multi-hour run survive a disconnect/kill without redoing work.
    done = set()
    if resume and out_fp.exists():
        with open(out_fp, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["episode_index"]))
                except Exception:  # noqa: BLE001
                    pass
        print(f"[stage] resume: {len(done)} episodes already done in "
              f"{out_fp}, skipping them", flush=True)
    mode = "a" if (resume and out_fp.exists()) else "w"
    with open(out_fp, mode, encoding="utf-8") as fo:
        for ep in eps:
            ei = int(ep["episode_index"])
            if ei in done:
                continue
            L = int(ep.get("length", 0))
            task = (ep.get("tasks") or [""])[0]
            mp4 = _video_path(dataset, ei, cam)
            if L < 2 or not mp4.exists():
                skip += 1
                continue
            idxs = np.linspace(0, L - 1, num=min(frames_k, L),
                               dtype=int).tolist()
            try:
                t0 = time.time()
                frs = _read_frames(mp4, idxs)
                print(f"[stage] ep{ei} L={L} -> VLM "
                      f"(mt={max_tokens})...", end="", flush=True)
                raw = _vlm_stage_call(base_url, model, api_key, task, L,
                                      idxs, frs, max_tokens=max_tokens,
                                      no_think=no_think, timeout=timeout,
                                      prefill=prefill)
                stages = _norm_stages(raw, L)
                print(f" {len(stages)} stages in {time.time()-t0:.1f}s",
                      flush=True)
                if debug and not dumped:
                    print(f"[stage][debug] ep{ei} raw (first 500):\n"
                          f"{raw[:500]}\n[stage][debug] -> {stages}")
                    dumped = True
            except Exception as e:  # noqa: BLE001
                print(" FAIL", flush=True)  # close the in-progress line
                # First failure: print the raw reply so the cause (empty
                # content / wrong shape / thinking) is visible, not hidden.
                if not dumped:
                    try:
                        _t, _r = _vlm_stage_call(
                            base_url, model, api_key, task, L, idxs,
                            _read_frames(mp4, idxs), max_tokens=max_tokens,
                            no_think=no_think, timeout=timeout,
                            return_raw=True, prefill=prefill)
                    except Exception as e2:  # noqa: BLE001
                        print(f"[stage][debug] ep{ei} raw call still "
                              f"failing: {e2}")
                    else:
                        print(f"[stage][debug] ep{ei} server replied but "
                              f"parse failed; text head:\n{_t[:500]}")
                    dumped = True
                print(f"[stage] ep{ei} SKIP: {e}")
                skip += 1
                continue
            fo.write(json.dumps({"episode_index": ei, "length": L,
                                 "tasks": ep.get("tasks"),
                                 "stages": stages}) + "\n")
            fo.flush()
            ok += 1
            nstage += len(stages)
            if ok % 20 == 0:
                print(f"[stage] {ok} done (avg_stages="
                      f"{nstage / max(ok,1):.2f})")
    print(f"[stage] {dataset.name}: ok={ok} skipped={skip} "
          f"avg_stages={nstage / max(ok,1):.2f} -> {out_fp}")


def main():
    ap = argparse.ArgumentParser(description="VLM offline stage annotation")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cam-key", default="observation.images.cam_high")
    ap.add_argument("--frames", type=int, default=6,
                    help="frames sampled per episode for the VLM")
    ap.add_argument("--episodes-file", default="episodes.jsonl")
    ap.add_argument("--out-name", default="stages.jsonl")
    ap.add_argument("--recursive", action="store_true",
                    help="annotate every task dir with meta/<episodes-file> "
                         "under --dataset (multi-task, follows symlinks)")
    ap.add_argument("--base-url",
                    default=_cred("VLLM_BASE_URL", HARDCODED_VLLM_BASE_URL))
    ap.add_argument("--model",
                    default=_cred("VLLM_MODEL", HARDCODED_VLLM_MODEL))
    ap.add_argument("--api-key", default=_cred("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="generation budget; this server essays before the "
                         "JSON so it needs room (default 4096)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-call HTTP timeout in seconds. This serve_qwen "
                         "is on a slow torch fallback path (~minutes/call); "
                         "too low -> silent retries that look like a hang "
                         "(default 600)")
    ap.add_argument("--think", action="store_true",
                    help="allow Qwen <think> reasoning (default: disabled "
                         "via /no_think + enable_thinking=False)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only annotate the first N episodes per task "
                         "(0=all; use a small N for a quick smoke test)")
    ap.add_argument("--debug", action="store_true",
                    help="print the first raw VLM reply per task dir")
    ap.add_argument("--prefill", action="store_true",
                    help="seed the assistant turn with '{\"stages\": [' so "
                         "the model continues JSON instead of narrating "
                         "(strongest anti-essay lever for this server)")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes already in stages.jsonl and APPEND "
                         "the rest (survives a disconnect/kill on the "
                         "multi-hour full run; without it the file is "
                         "overwritten from scratch)")
    ap.add_argument("--probe", action="store_true",
                    help="diagnostic: dump the raw server reply for ep0 of "
                         "the (first) task and exit; writes nothing")
    args = ap.parse_args()
    no_think = not args.think

    root = Path(args.dataset)
    if args.probe:
        if args.recursive:
            t = sorted({
                Path(dp) for dp, _d, _f in os.walk(root, followlinks=True)
                if (Path(dp) / "meta" / args.episodes_file).exists()})
            if not t:
                raise SystemExit(f"--probe: no task dir under {root}")
            root = t[0]
        probe(root, args.cam_key, args.frames, args.episodes_file,
              args.base_url, args.model, args.api_key, args.max_tokens,
              no_think, args.timeout, args.prefill)
        return
    if args.recursive:
        targets = sorted({
            Path(dp) for dp, _d, _f in os.walk(root, followlinks=True)
            if (Path(dp) / "meta" / args.episodes_file).exists()})
        if not targets:
            raise SystemExit(f"--recursive: no task dir under {root}")
        print(f"[stage] recursive: {len(targets)} task dirs")
        for i, t in enumerate(targets):
            print(f"[stage] ({i+1}/{len(targets)}) {t}")
            try:
                annotate(t, args.cam_key, args.frames, args.episodes_file,
                         args.out_name, args.base_url, args.model,
                         args.api_key, args.max_tokens, no_think,
                         args.limit, args.debug, args.timeout,
                         args.prefill, args.resume)
            except Exception as e:  # noqa: BLE001
                print(f"[stage] SKIP {t}: {e}")
    else:
        annotate(root, args.cam_key, args.frames, args.episodes_file,
                 args.out_name, args.base_url, args.model, args.api_key,
                 args.max_tokens, no_think, args.limit, args.debug,
                 args.timeout, args.prefill, args.resume)


if __name__ == "__main__":
    main()
