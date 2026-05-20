# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""VLM 任务完成度评判 (per-subgoal completion judge).

读取 inference log (每个任务一个 *.log,格式见 outputs_infonce/log/<task>/),
解析每集的 prompt + 有序 subgoals + real_video 路径;对 real_video 均匀抽 K
帧,送 Qwen3-VL-4B-Instruct (OpenAI 兼容端点,默认 106.12.146.172:8271/v1,
参见 qwen_api.py) 做"逐子目标完成度"评分,写 <out-dir>/<task>.judge.jsonl。

输入 log 中每条 episode 的格式 (示例):
  [Episode 0] idx=0 succ=True steps=110 real_video=... imagary_video=... latent=...
    prompt: 'Beat the block after grabbing the nail-driving hammer'
    subgoals:
      [0] {'name': 'Grab Hammer', 'goal': 'Pick up the nail-driving hammer ...'}
      [1] {'name': 'Move to Block', 'goal': 'Position the hammer above the red block'}
      [2] {'name': 'Strike Block', 'goal': 'Hit the red block with the hammer'}

输出每行 JSON:
  {episode_index, logged_success, logged_steps, prompt, real_video,
   subgoals:[{index,name,goal,completion(0-1),evidence}],
   overall_completion(0-1), notes, vlm_elapsed_s, vlm_usage}

依赖:openai, httpx, imageio[ffmpeg], Pillow, numpy
    pip install openai httpx imageio imageio-ffmpeg Pillow numpy
"""
from __future__ import annotations

import argparse
import ast
import base64
import glob
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# imageio / httpx / openai 都做惰性 import,允许 parse_log/_parse_json_robust
# 在没装这些依赖的环境里被单元测试。read_frames/call_vlm 才真正需要。


# --- log 解析 ---------------------------------------------------------------

_EP_HDR = re.compile(
    r"^\[Episode\s+(\d+)\]\s+idx=(\d+)\s+succ=(True|False)\s+steps=(\d+)\s+"
    r"real_video=(\S+)\s+imagary_video=(\S+)\s+latent=(\S+)\s*$"
)
_SUB_LINE = re.compile(r"^\s*\[(\d+)\]\s+(\{.*\})\s*$")


def parse_log(fp: Path):
    """读一个任务 .log -> [dict]。容忍空行、单/双引号、不齐缩进。"""
    episodes, cur = [], None
    with open(fp, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _EP_HDR.match(line)
            if m:
                if cur is not None:
                    episodes.append(cur)
                cur = {
                    "episode": int(m.group(1)),
                    "idx": int(m.group(2)),
                    "succ": m.group(3) == "True",
                    "steps": int(m.group(4)),
                    "real_video": m.group(5),
                    "imagary_video": m.group(6),
                    "latent": m.group(7),
                    "prompt": "",
                    "subgoals": [],
                }
                continue
            if cur is None:
                continue
            s = line.strip()
            if s.startswith("prompt:"):
                p = s[len("prompt:"):].strip()
                if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
                    p = p[1:-1]
                cur["prompt"] = p
                continue
            ms = _SUB_LINE.match(line)
            if ms:
                try:
                    d = ast.literal_eval(ms.group(2))
                    cur["subgoals"].append({
                        "name": str(d.get("name", "")).strip(),
                        "goal": str(d.get("goal", "")).strip(),
                    })
                except Exception:  # noqa: BLE001
                    pass
    if cur is not None:
        episodes.append(cur)
    return episodes


# --- 视频解码 + 编码为 data url ---------------------------------------------

def read_frames(mp4: str, k: int):
    """均匀采 k 帧;imageio ffmpeg 优先,失败回退 pyav。"""
    try:
        import imageio.v2 as iio
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "需要 imageio + imageio-ffmpeg: "
            f"pip install imageio imageio-ffmpeg ({e})")
    last = None
    try:
        rdr = iio.get_reader(str(mp4), "ffmpeg")
        meta = rdr.get_meta_data() or {}
        n = meta.get("nframes")
        if not n or n == float("inf"):
            frs = []
            for fr in rdr:
                frs.append(np.asarray(fr))
                if len(frs) >= 4096:
                    break
            rdr.close()
            n = len(frs)
            if n == 0:
                raise RuntimeError("zero frames")
            idxs = np.linspace(0, n - 1, num=min(k, n), dtype=int).tolist()
            return [frs[i] for i in idxs]
        n = int(n)
        idxs = np.linspace(0, max(n - 1, 0), num=min(k, n),
                           dtype=int).tolist()
        out = []
        for i in idxs:
            try:
                out.append(np.asarray(rdr.get_data(int(i))))
            except Exception:  # noqa: BLE001
                out.append(np.asarray(rdr.get_data(0)))
        rdr.close()
        return out
    except Exception as e:  # noqa: BLE001
        last = e
    try:
        import imageio.v3 as iio3
        frames = iio3.imread(str(mp4), plugin="pyav")  # [T,H,W,C]
        n = len(frames)
        if n == 0:
            raise RuntimeError("zero frames (pyav)")
        idxs = np.linspace(0, n - 1, num=min(k, n), dtype=int).tolist()
        return [np.asarray(frames[i]) for i in idxs]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot decode {mp4} ({last}; pyav: {e})")


def img_to_data_url(arr, max_side: int = 512, quality: int = 85) -> str:
    """np.uint8 RGB -> data:image/jpeg;base64,..."""
    from PIL import Image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    img = Image.fromarray(arr)
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# --- VLM 调用 ---------------------------------------------------------------

_SYS = """You are a strict robot task completion judge.
You receive frames sampled in chronological order from a robot's ACTUAL
execution video, the natural-language task instruction, and a list of
ORDERED planned subgoals. For EACH subgoal, decide how completely it was
achieved using ONLY the visible visual evidence; then judge overall task
completion.

Scoring rubric for completion (float in [0,1]):
  0.00  not attempted / no visible progress
  0.30  attempted but clearly failed (e.g. missed grasp, dropped)
  0.60  partially achieved (some progress but not fully done)
  1.00  fully achieved (the visible outcome matches the subgoal goal)

ABSOLUTE OUTPUT RULES (a parser reads your reply with json.loads):
- Output ONE JSON object and NOTHING else. No prose, no markdown, no
  reasoning, no "Step 1:" enumeration.
- The very first character of your reply MUST be '{' and the very last
  MUST be '}'.
- Schema EXACTLY:
  {"subgoals":[{"index":<int>,"completion":<float 0..1>,
                "evidence":"<<=20 words>"}],
   "overall_completion":<float 0..1>,
   "notes":"<<=30 words>"}
- subgoals MUST cover every input subgoal index in order (0..N-1).
"""


def _build_user(prompt: str, subgoals, frames):
    sub_text = "\n".join(
        f"  [{i}] name: {s['name']}\n       goal: {s['goal']}"
        for i, s in enumerate(subgoals))
    user_text = (
        f"TASK INSTRUCTION: {prompt}\n"
        f"PLANNED ORDERED SUBGOALS ({len(subgoals)} items):\n{sub_text}\n\n"
        f"The {len(frames)} images that follow are uniformly-sampled frames "
        "from the robot's ACTUAL execution video, in chronological order. "
        "Judge per-subgoal completion from what is visible. "
        "Output JSON only (no prose, no markdown)."
    )
    content = [{"type": "text", "text": user_text}]
    for fr in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": img_to_data_url(fr)}})
    return content


def _parse_json_robust(txt: str):
    """从可能含散文 / fences / 截断的回复里把 JSON 挖出来。"""
    s = (txt or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    a = s.find("{")
    if a < 0:
        raise ValueError(f"no JSON object in reply: {s[:200]!r}")
    # brace/quote-aware 扫描末尾匹配 } (或截断时补齐)
    depth, instr, esc = 0, False, False
    end = -1
    for i in range(a, len(s)):
        c = s[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        return json.loads(s[a:end])
    frag = s[a:] + ('"' if instr else '') + ("}" * max(depth, 1))
    try:
        return json.loads(frag)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"unbalanced JSON: {s[:200]!r} ({e})")


def call_vlm(client, model, prompt, subgoals, frames, max_tokens, timeout):
    """单次 VLM 调用 -> (parsed_obj, raw_text, usage_dict)。重试 4 次。"""
    content = _build_user(prompt, subgoals, frames)
    last = None
    for a in range(4):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": _SYS},
                          {"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=int(max_tokens),
                stream=False,
                timeout=timeout,
            )
            # 头一次尝试加 response_format(若端点不识别会 4xx,后续重试不加)
            if a == 0:
                kwargs["extra_body"] = {
                    "response_format": {"type": "json_object"}}
            r = client.chat.completions.create(**kwargs)
            txt = (r.choices[0].message.content or "").strip()
            if not txt:
                raise ValueError("empty content")
            usage = (r.usage.model_dump()
                     if hasattr(r, "usage") and r.usage else None)
            return _parse_json_robust(txt), txt, usage
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** a, 6))
    raise RuntimeError(f"VLM call failed after 4 tries: {last}")


# --- 主流程 -----------------------------------------------------------------

def judge_task(log_fp: Path, video_root: str, out_fp: Path, model: str,
               base_url: str, api_key: str, frames_k: int, max_tokens: int,
               timeout: int, limit: int, resume: bool):
    eps = parse_log(log_fp)
    if limit > 0:
        eps = eps[:limit]

    done = set()
    if resume and out_fp.exists():
        kept = {}
        for line in open(out_fp, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                kept[int(rec["episode_index"])] = rec
            except Exception:  # noqa: BLE001
                pass
        # 清洗去重 + 截断保护:重写有效行,再 append
        with open(out_fp, "w", encoding="utf-8") as fw:
            for ei in sorted(kept):
                fw.write(json.dumps(kept[ei], ensure_ascii=False) + "\n")
        done = set(kept)
        print(f"[judge] resume: {len(done)} valid episodes kept in "
              f"{out_fp.name}, skipping them")

    mode = "a" if (resume and out_fp.exists()) else "w"
    import httpx
    from openai import OpenAI
    http_client = httpx.Client(trust_env=False)
    client = OpenAI(api_key=api_key, base_url=base_url,
                    http_client=http_client)

    ok = skip = 0
    nsub_pass = ntot = 0
    overall_sum = 0.0
    with open(out_fp, mode, encoding="utf-8") as fo:
        for ep in eps:
            ei = ep["episode"]
            if ei in done:
                continue
            if not ep["subgoals"]:
                print(f"[judge] ep{ei} no subgoals, skip")
                skip += 1
                continue
            vp = ep["real_video"]
            if not os.path.isabs(vp):
                vp = os.path.join(video_root, vp)
            if not os.path.exists(vp):
                print(f"[judge] ep{ei} video not found: {vp}")
                skip += 1
                continue
            try:
                t0 = time.time()
                frs = read_frames(vp, frames_k)
                obj, raw, usage = call_vlm(
                    client, model, ep["prompt"], ep["subgoals"], frs,
                    max_tokens, timeout)
                dt = time.time() - t0
            except Exception as e:  # noqa: BLE001
                print(f"[judge] ep{ei} FAIL: {e}")
                skip += 1
                continue

            # 规范化 subgoals 输出:与输入逐位对齐(允许 VLM 漏一两条/乱序)
            subs_in = ep["subgoals"]
            subs_out = obj.get("subgoals") or []
            by_idx = {}
            for r in subs_out:
                try:
                    by_idx[int(r.get("index", -1))] = r
                except Exception:  # noqa: BLE001
                    pass
            norm = []
            for i, s in enumerate(subs_in):
                r = by_idx.get(i)
                if r is None and i < len(subs_out):
                    r = subs_out[i]
                try:
                    comp = float(r.get("completion", 0.0)) if r else 0.0
                except Exception:  # noqa: BLE001
                    comp = 0.0
                comp = max(0.0, min(1.0, comp))
                ev = (str(r.get("evidence", ""))[:200] if r else "")
                norm.append({"index": i, "name": s["name"], "goal": s["goal"],
                             "completion": comp, "evidence": ev})

            try:
                oc = float(obj.get("overall_completion",
                                   sum(n["completion"] for n in norm)
                                   / max(len(norm), 1)))
            except Exception:  # noqa: BLE001
                oc = sum(n["completion"] for n in norm) / max(len(norm), 1)
            oc = max(0.0, min(1.0, oc))

            rec = {
                "episode_index": ei,
                "logged_success": ep["succ"],
                "logged_steps": ep["steps"],
                "prompt": ep["prompt"],
                "real_video": vp,
                "subgoals": norm,
                "overall_completion": oc,
                "notes": str(obj.get("notes", ""))[:300],
                "vlm_elapsed_s": round(dt, 2),
                "vlm_usage": usage,
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            ok += 1
            nsub_pass += sum(1 for n in norm if n["completion"] >= 0.6)
            ntot += len(norm)
            overall_sum += oc
            print(f"[judge] {log_fp.stem} ep{ei}: overall={oc:.2f} "
                  f"subs={[round(n['completion'], 2) for n in norm]} "
                  f"{dt:.1f}s "
                  + ("[succ]" if ep["succ"] else "[fail]"))

    if ok:
        print(f"[judge] {log_fp.name} done: ok={ok} skip={skip} "
              f"mean_overall={overall_sum / ok:.3f} "
              f"sub_pass_rate@0.6={nsub_pass / max(ntot, 1):.3f}")
    else:
        print(f"[judge] {log_fp.name}: no new episodes judged "
              f"(done={len(done)} skip={skip})")


def main():
    ap = argparse.ArgumentParser(description="VLM 任务完成度评判")
    ap.add_argument("--log-root", required=True,
                    help="包含 <task>/<*.log> 的目录,例如 "
                         "/inspire/qb-ilm2/.../RoboTwin/outputs_infonce/log")
    ap.add_argument("--video-root",
                    default="/inspire/qb-ilm2/project/26summer-camp-11/"
                            "public/group3/RoboTwin",
                    help="log 中 real_video 的相对路径所参照的根目录")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录;默认 <log-root>/judge")
    ap.add_argument("--task", default=None,
                    help="只跑指定任务子目录;不指定 = 跑全部")
    ap.add_argument("--frames", type=int, default=8,
                    help="每段视频均匀采样的帧数(default 8)")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0,
                    help="每任务只跑前 N 集(0=全部)")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 <task>.judge.jsonl 已评判的 episode 并追加")
    ap.add_argument("--base-url", default="http://106.12.146.172:8271/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir) if args.out_dir else log_root / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task:
        logs = sorted((log_root / args.task).glob("*.log"))
    else:
        logs = sorted(Path(p) for p in glob.glob(
            str(log_root / "*" / "*.log")))
    if not logs:
        raise SystemExit(f"no .log files under {log_root}"
                         + (f"/{args.task}" if args.task else ""))

    print(f"[judge] {len(logs)} log file(s); out_dir={out_dir}")
    print(f"[judge] VLM: model={args.model} base_url={args.base_url}")
    for lf in logs:
        task = lf.parent.name
        out_fp = out_dir / f"{task}.judge.jsonl"
        print(f"\n=========== {task} ({lf.name}) -> {out_fp.name} ===========")
        try:
            judge_task(lf, args.video_root, out_fp, args.model,
                       args.base_url, args.api_key,
                       frames_k=args.frames, max_tokens=args.max_tokens,
                       timeout=args.timeout, limit=args.limit,
                       resume=args.resume)
        except Exception as e:  # noqa: BLE001
            print(f"[judge] SKIP {lf}: {e}")

    # 全任务汇总
    agg = {}
    for lf in logs:
        task = lf.parent.name
        fp = out_dir / f"{task}.judge.jsonl"
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in open(fp, "r", encoding="utf-8")
                if l.strip()]
        if not rows:
            continue
        nsub = sum(len(r["subgoals"]) for r in rows)
        nsub_pass = sum(sum(1 for s in r["subgoals"] if s["completion"] >= 0.6)
                        for r in rows)
        agg[task] = {
            "n_episodes_judged": len(rows),
            "mean_overall_completion": round(
                sum(r["overall_completion"] for r in rows) / len(rows), 3),
            "sub_pass_rate@0.6": round(nsub_pass / max(nsub, 1), 3),
            "logged_success_rate": round(
                sum(1 for r in rows if r["logged_success"]) / len(rows), 3),
            "n_subgoals_total": nsub,
        }
    print("\n========== AGGREGATE ==========")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"\n[judge] summary -> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
