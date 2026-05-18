# WAM-CoT on Robocasa (26-summer project)

Adapts **LingBot-VA** to the **Robocasa** simulator and implements the PDF's
**Phase-2, Route-1 (External Semantic CoT)**: a DeepSeek VLM acts as a
high-level planner that turns a task instruction + camera frame into an
*explicit physical-constraint reasoning trace* and an *ordered list of atomic
sub-tasks*, which the low-level world-action model executes one by one.

```
Robocasa frame ─► DeepSeek PLANNER ─► [sub-task₁, sub-task₂, …]  (+ reasoning)
                                          │
   reset(prompt=sub-task₁) ─► WAM rollout │ every K key-frames:
                                          ▼
                                   DeepSeek MONITOR
                       subtask_done ─► switch_prompt(sub-task₂)   ← soft switch
                       need_replan  ─► PLANNER again              (KV/world-model
                       task_success ─► stop                        context kept)
```

## 1. Architecture & the zero-shot decision

We evaluate **zero-shot** with the released `lingbot-va-posttrain-libero-long`
checkpoint (no Robocasa fine-tuning). LIBERO and Robocasa are both
robosuite/single-arm, so the **model-facing interface is kept byte-identical
to LIBERO**: 2 cameras (`agentview_rgb`, `eye_in_hand_rgb`) @128², 7-dim OSC
action, LIBERO quantile norm-stats (see `wan_va/configs/va_robocasa_cfg.py`).
**All** Robocasa adaptation is client-side in this folder:

| file | role |
|---|---|
| `robocasa_env.py` | env wrapper: Robocasa obs→2 LIBERO keys, 7-dim action→composite-controller action, success, language |
| `probe_env.py` | **run first** — dumps the real Robocasa API so mappings can be locked |
| `eval_common.py` | WAM websocket session + LIBERO-identical action-chunk/KV loop + video/metrics |
| `client.py` | Phase-1 **Baseline** WAM (no reasoning) |
| `cot_planner.py` | DeepSeek planner (`plan`) + VLM monitor (`monitor`), OpenAI-compatible, call logging |
| `client_cot.py` | **WAM-CoT** Route-1 orchestration + ablation matrix |
| `calc_stat.py` | baseline vs CoT vs ablations → CSV + charts + report.md |

Server side: one minimal, additive change in `wan_va/wan_va_server.py` — a
`switch_prompt` message + `_switch_prompt()` that re-encodes the sub-task
language **without** clearing the KV cache / streaming-VAE / `frame_st_id`, so
the autoregressive world-model context carries across sub-task boundaries.
`wan_va/configs/va_robocasa_cfg.py` registered as config name `robocasa`.

> ⚠️ **Expectation:** zero-shot LIBERO→Robocasa has a real visual + action-space
> domain gap, so Baseline SR will likely be low. That is exactly the
> "Baseline 瓶颈" the report must analyse; WAM-CoT is judged on the *relative*
> long-horizon/occlusion improvement and the ablations, not absolute SR.

## 2. Setup (on the internet-capable 4090)

Everything (Robocasa + LingBot server + DeepSeek planner) runs on the 4090 so
the planner can reach `api.deepseek.com`.

> ### WHERE to run the commands
> The LingBot repo on the **server** is:
> ```
> /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va
> ```
> (your local `E:\sii_program\sii_wam_cot` is just the editing copy — sync the
> new/changed files there first; see §0). **`cd` into the server repo root and
> run everything from there** — that is what your failed
> `python -m evaluation.robocasa.probe_env` was missing (you were in the
> robocasa asset dir, which has no `evaluation` package). The scripts use the
> **cwd-independent script-path form** (`python evaluation/robocasa/X.py`), so
> they work as long as you launched them from the repo root. `robocasa` is
> pip-installed in the conda env, so `import robocasa` resolves from any cwd.

```bash
LINGBOT=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va

# (a) checkpoint: va_robocasa_cfg.py already points at
#     $LINGBOT/checkpoints/lingbot-va-posttrain-libero-long
#     -> just set that ckpt's transformer/config.json "attn_mode": "torch"

# (b) DeepSeek creds: edit ONCE in code (no `export` ever needed) —
#     evaluation/robocasa/cot_planner.py : HARDCODED_DEEPSEEK_API_KEY
#     (also HARDCODED_DEEPSEEK_MODEL if "deepseek-v4-pro" isn't your id)
```

## 0. Sync local edits to the server repo

These files are new/changed and must exist under `$LINGBOT`:
`wan_va/wan_va_server.py`, `wan_va/configs/__init__.py`,
`wan_va/configs/va_robocasa_cfg.py`, `evaluation/__init__.py`, and the whole
`evaluation/robocasa/` directory.

## 3. Probe the real env FIRST (important)

The Robocasa install lives on the server and its robosuite version / camera
names / composite-controller action layout cannot be seen from the dev box.
Run from the **server repo root** (script-path form, cwd-independent):

```bash
conda activate robocasa
cd $LINGBOT
python evaluation/robocasa/probe_env.py --env PnPCounterToCab \
       --save-frames --out outputs/robocasa_probe.json
```

Check in the JSON: `image_keys`, `action_dim`, `action_split_indexes`,
`task_language`. If they differ from the PandaOmron defaults, pass corrections
via `ENV_OVERRIDES` (a JSON dict overriding `RobocasaConfig`), e.g.:

```bash
export ENV_OVERRIDES='{"camera_map":{"robot0_agentview_center":"observation.images.agentview_rgb","robot0_eye_in_hand":"observation.images.eye_in_hand_rgb"},"arm_action_slice":[0,6],"gripper_index":6,"robots":"PandaOmron"}'
```

`robocasa_env.py` auto-detects most of this at runtime; overrides are only the
escape hatch when auto-detect is wrong.

## 4. Run

```bash
cd $LINGBOT

# terminal A — lingbot conda env: inference server
bash evaluation/robocasa/launch_server.sh

# terminal B — robocasa conda env, from $LINGBOT: baseline
bash evaluation/robocasa/launch_client.sh

# terminal B: full WAM-CoT
ABLATION=none bash evaluation/robocasa/launch_client_cot.sh

# everything (baseline + CoT + 必做 ablations) + report:
bash evaluation/robocasa/run_ablations.sh
```

Pick tasks from `python -m robocasa.demos.demo_tasks`; set via
`TASKS="EnvA EnvB"`. Defaults target long-horizon / occlusion-prone tasks
(`PnPCounterToCab`, `PnPCounterToMicrowave`, `OpenDrawer`, `PnPCounterToSink`).

## 5. Ablations (PDF 消融实验 · 必做)

`--ablation` / `ABLATION=`:

| value | what it removes | tests |
|---|---|---|
| `none` | — | full WAM-CoT |
| `no_cot` | the planner entirely (single full-task prompt) | == Baseline within the same harness (fair A/B) |
| `shuffle_subtasks` | correct sub-task **order** | value of ordered decomposition |
| `no_monitor` | VLM feedback (advance only on step budget) | value of the CoT **observation** model (退化) |
| `blind_planner` | the image into the planner (text-only) | value of visual grounding in planning |
| `hard_reset` | the soft prompt-switch (uses `reset` instead) | value of world-model **context carryover** |

## 6. Outputs (report deliverables)

Per run dir: `<task>.json` (cumulative SR / sub-task progress / avg steps /
failure-tag histogram / VLM cost), per-episode
`<ep>_<success>_<failuretag>.mp4` (2-cam video; CoT runs burn the sub-task
timeline on top), `<ep>_*.plan.json` (reasoning + sub-task plan + monitor
events — the interpretability artefact), `vlm_calls.jsonl` (every DeepSeek
call: latency + tokens). `calc_stat.py` → `comparison.csv`,
`sr_comparison.png`, `failure_breakdown.png`, `report.md`.

Failure taxonomy (`eval_common.classify_failure`): `success`,
`timeout_no_progress`, `wrong_object_or_location`, `grasp_failure`,
`subtask_stuck`, `planner_error`, `env_error` — feeds the report's
失败案例机制分析.

## 7. Analysis hooks for the report

- **Baseline 瓶颈**: inspect `wrong_object_or_location` / `timeout_no_progress`
  on long-horizon tasks; the `.plan.json` reasoning shows what decomposition
  the baseline lacks.
- **CoT 贡献**: `none` vs `no_cot` SR, and especially `subtask_progress_rate`.
- **机制可解释性**: the `.plan.json` reasoning + the on-video sub-task timeline
  show the CoT is actually steering action generation.
- **CoT 真实参与**: `hard_reset` vs `none` isolates the soft-switch /
  world-model-context contribution; `shuffle_subtasks` confirms ordering (not
  just "more prompts") drives the gain.
- **开销**: `vlm_calls.jsonl` / `report.md` give the 推理时间与计算开销统计.
