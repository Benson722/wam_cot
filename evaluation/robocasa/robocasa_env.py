# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Robocasa <-> LingBot-VA adapter.

This module is the *only* place that knows about Robocasa / robosuite. It
exposes a thin :class:`RobocasaEnv` wrapper that:

  1. builds a Robocasa kitchen environment (robosuite-based),
  2. converts Robocasa observations into the exact 2-camera LIBERO-style
     observation dict the zero-shot ``lingbot-va-posttrain-libero-long``
     checkpoint expects (``observation.images.agentview_rgb`` /
     ``observation.images.eye_in_hand_rgb``, uint8 HxWx3),
  3. expands the model's 7-dim OSC action
     ``[dx, dy, dz, drx, dry, drz, gripper]`` into Robocasa's full
     composite-controller action vector (arm + gripper filled, base / torso
     zeroed), and
  4. reports task success and the natural-language task instruction.

Because the Robocasa install lives on the remote server and its exact
robosuite version / controller layout / camera names cannot be inspected
from here, every environment-specific quantity is **auto-detected at runtime
and overridable**. Run ``python -m evaluation.robocasa.probe_env`` on the
server first; paste its JSON back and (if needed) override the few fields in
:class:`RobocasaConfig` that auto-detection got wrong.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("robocasa_env")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RobocasaConfig:
    """All Robocasa-side knobs. Sensible PandaOmron defaults; the probe script
    tells you which (if any) need overriding for your robosuite version."""

    # --- task / robot -----------------------------------------------------
    env_name: str = "PickPlaceCounterToCabinet"
    robots: str = "PandaOmron"
    # Controller: None -> let helper pick the Robocasa default composite
    # controller ("BASIC"). Override with a robosuite controller name/json.
    controller: Optional[str] = None
    layout_ids: Optional[int] = None          # None -> random kitchen layout
    style_ids: Optional[int] = None           # None -> random kitchen style
    translucent_robot: bool = False
    seed: Optional[int] = None

    # --- rendering --------------------------------------------------------
    # Rendered at this resolution; the LingBot server downsamples to 128 anyway,
    # but a slightly larger render keeps the saved videos legible.
    camera_height: int = 256
    camera_width: int = 256
    control_freq: int = 20

    # Robocasa camera name -> LingBot key. The *value* names are fixed (the
    # checkpoint was trained on them); only change the *key* (left side) if the
    # probe shows different Robocasa camera names for your robot/version.
    camera_map: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "robot0_agentview_left": "observation.images.agentview_rgb",
            "robot0_eye_in_hand": "observation.images.eye_in_hand_rgb",
        }
    )
    # Fallback agentview candidates tried (in order) if the mapped key is absent.
    agentview_fallbacks: Tuple[str, ...] = (
        "robot0_agentview_center",
        "robot0_agentview_right",
        "agentview",
        "robot0_frontview",
        "frontview",
    )
    eye_in_hand_fallbacks: Tuple[str, ...] = ("robot0_eye_in_hand", "eye_in_hand")
    # robosuite renders camera images upside-down (origin bottom-left); the
    # LIBERO client flips with [::-1]. Keep True unless the probe preview looks
    # flipped the other way.
    flip_images_vertically: bool = True

    # --- action layout (7-dim model action -> full env action) ------------
    # Auto-detected from robot.composite_controller split indexes when
    # possible. These are PandaOmron "BASIC" defaults used as fallback.
    #   arm_action_slice : where the 6 OSC pose deltas go
    #   gripper_index    : single gripper scalar position
    # Everything else in the action vector is set to 0 (base/torso stationary).
    action_dim: Optional[int] = None          # None -> read from env.action_dim
    arm_action_slice: Tuple[int, int] = (0, 6)
    gripper_index: int = 6
    # Scaling / sign to bridge the LIBERO<->Robocasa convention gap. These are
    # the main zero-shot knobs to sweep when analysing the baseline bottleneck.
    arm_action_scale: float = 1.0
    gripper_scale: float = 1.0
    gripper_sign: float = 1.0                  # flip to -1.0 if gripper inverted
    clip_to_action_spec: bool = True

    # --- episode ----------------------------------------------------------
    max_env_steps: int = 1200                  # hard cap per episode

    def merge_overrides(self, overrides: Optional[Dict[str, Any]]) -> "RobocasaConfig":
        if not overrides:
            return self
        cfg = dataclasses.replace(self)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                logger.warning("Ignoring unknown RobocasaConfig override: %s", k)
                continue
            setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Robosuite/Robocasa import + controller helpers (version tolerant)
# ---------------------------------------------------------------------------
def _import_robosuite():
    try:
        import robocasa  # noqa: F401  (registers kitchen envs into robosuite)
    except Exception as e:  # pragma: no cover - server-only
        raise ImportError(
            "Failed to `import robocasa`. Activate the `robocasa` conda env and "
            "ensure robocasa_suite is on PYTHONPATH (see evaluation/robocasa/README.md)."
        ) from e
    import robosuite  # noqa: F401

    return robosuite


def _make_controller_config(robosuite, robots: str, controller: Optional[str]):
    """robosuite >=1.5 uses composite controllers; older uses single. Try both."""
    # Newer robosuite / Robocasa path
    try:
        from robosuite.controllers import load_composite_controller_config

        return load_composite_controller_config(controller=controller, robot=robots)
    except Exception:  # pragma: no cover - depends on server version
        pass
    # Older robosuite path
    try:
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller=controller or "OSC_POSE")
    except Exception as e:  # pragma: no cover
        logger.warning("Could not build a controller config (%s); using None.", e)
        return None


def _env_camera_names(env) -> List[str]:
    for attr in ("camera_names", "_camera_names"):
        v = getattr(env, attr, None)
        if v:
            return list(v)
    return []


def _detect_action_layout(env, cfg: RobocasaConfig) -> RobocasaConfig:
    """Best-effort read of arm/gripper action indices from the robot's
    composite controller; falls back to the cfg defaults."""
    out = dataclasses.replace(cfg)
    try:
        out.action_dim = int(env.action_dim)
    except Exception:
        out.action_dim = cfg.action_dim
    try:
        robot = env.robots[0]
        split = getattr(robot, "_action_split_indexes", None) or getattr(
            robot, "action_split_indexes", None
        )
        # split: dict part_name -> (start, end)
        if isinstance(split, dict):
            arm_key = next(
                (k for k in split if "right" in k.lower() and "grip" not in k.lower()),
                None,
            ) or next((k for k in split if "arm" in k.lower()), None)
            grip_key = next((k for k in split if "grip" in k.lower()), None)
            if arm_key is not None:
                s, e = split[arm_key]
                out.arm_action_slice = (int(s), int(s) + 6)
            if grip_key is not None:
                gs, _ = split[grip_key]
                out.gripper_index = int(gs)
            logger.info(
                "Detected action layout from controller: arm=%s gripper=%s (action_dim=%s)",
                out.arm_action_slice,
                out.gripper_index,
                out.action_dim,
            )
    except Exception as e:
        logger.warning(
            "Action-layout auto-detect failed (%s); using cfg defaults arm=%s grip=%s",
            e,
            cfg.arm_action_slice,
            cfg.gripper_index,
        )
    return out


# ---------------------------------------------------------------------------
# The environment wrapper
# ---------------------------------------------------------------------------
class RobocasaEnv:
    """LingBot-facing Robocasa wrapper. One instance == one episode lifecycle
    (call :meth:`reset` between episodes, like robosuite)."""

    def __init__(self, cfg: RobocasaConfig):
        self.cfg = cfg
        robosuite = _import_robosuite()
        self._robosuite = robosuite

        env_kwargs: Dict[str, Any] = dict(
            env_name=cfg.env_name,
            robots=cfg.robots,
            controller_configs=_make_controller_config(
                robosuite, cfg.robots, cfg.controller
            ),
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera=None,
            use_camera_obs=True,
            use_object_obs=True,
            camera_depths=False,
            camera_heights=cfg.camera_height,
            camera_widths=cfg.camera_width,
            control_freq=cfg.control_freq,
            ignore_done=True,
            translucent_robot=cfg.translucent_robot,
        )
        # Render every camera we might need (mapped + fallbacks).
        cams = list(cfg.camera_map.keys()) + list(cfg.agentview_fallbacks) + list(
            cfg.eye_in_hand_fallbacks
        )
        env_kwargs["camera_names"] = sorted(set(cams))
        if cfg.layout_ids is not None:
            env_kwargs["layout_ids"] = cfg.layout_ids
        if cfg.style_ids is not None:
            env_kwargs["style_ids"] = cfg.style_ids
        if cfg.seed is not None:
            env_kwargs["seed"] = cfg.seed

        # Some robosuite versions reject unknown kwargs; drop the optional ones
        # progressively until make() succeeds.
        self.env = self._make_env_resilient(robosuite, env_kwargs)
        self.cfg = _detect_action_layout(self.env, cfg)

        self._resolved_cam: Dict[str, str] = {}
        self._last_obs: Optional[Dict[str, Any]] = None
        self._ep_steps = 0
        self._task_language = ""

    # -- construction helpers ---------------------------------------------
    def _make_env_resilient(self, robosuite, env_kwargs: Dict[str, Any]):
        optional = [
            "translucent_robot",
            "render_camera",
            "ignore_done",
            "camera_depths",
            "use_object_obs",
        ]
        kwargs = dict(env_kwargs)
        last_err: Optional[Exception] = None
        for _ in range(len(optional) + 1):
            try:
                return robosuite.make(**kwargs)
            except TypeError as e:
                last_err = e
                # remove one offending optional kwarg and retry
                removed = False
                for opt in optional:
                    if opt in kwargs:
                        logger.warning(
                            "robosuite.make rejected kwargs; dropping '%s' and retrying.",
                            opt,
                        )
                        kwargs.pop(opt)
                        removed = True
                        break
                if not removed:
                    break
        raise RuntimeError(
            f"robosuite.make failed for env '{env_kwargs.get('env_name')}'. "
            f"Last error: {last_err}. Run probe_env.py to inspect the API."
        )

    # -- camera / observation conversion ----------------------------------
    def _resolve_camera_keys(self, raw_obs: Dict[str, Any]) -> None:
        """Pick concrete robosuite obs keys for the 2 LingBot views, once."""
        cfg = self.cfg

        def first_present(*candidates: str) -> Optional[str]:
            for c in candidates:
                if f"{c}_image" in raw_obs:
                    return f"{c}_image"
            return None

        # agentview
        agent_src = None
        for cam, lkey in cfg.camera_map.items():
            if lkey.endswith("agentview_rgb"):
                agent_src = first_present(cam)
                break
        if agent_src is None:
            agent_src = first_present(*cfg.agentview_fallbacks)
        # eye-in-hand
        eih_src = None
        for cam, lkey in cfg.camera_map.items():
            if lkey.endswith("eye_in_hand_rgb"):
                eih_src = first_present(cam)
                break
        if eih_src is None:
            eih_src = first_present(*cfg.eye_in_hand_fallbacks)

        if agent_src is None or eih_src is None:
            avail = sorted(k for k in raw_obs if k.endswith("_image"))
            raise KeyError(
                "Could not resolve LingBot camera views from Robocasa obs. "
                f"agentview={agent_src}, eye_in_hand={eih_src}. "
                f"Available image keys: {avail}. "
                "Fix RobocasaConfig.camera_map / fallbacks (see probe_env.py)."
            )
        self._resolved_cam = {
            "observation.images.agentview_rgb": agent_src,
            "observation.images.eye_in_hand_rgb": eih_src,
        }
        logger.info("Resolved cameras: %s", self._resolved_cam)

    def _img(self, raw_obs: Dict[str, Any], src_key: str) -> np.ndarray:
        im = raw_obs[src_key]
        im = np.asarray(im)
        if im.dtype != np.uint8:
            # robosuite normally returns uint8; handle float [0,1] just in case
            im = np.clip(im * (255.0 if im.max() <= 1.0 else 1.0), 0, 255).astype(
                np.uint8
            )
        if self.cfg.flip_images_vertically:
            im = im[::-1]
        return np.ascontiguousarray(im[..., :3])

    def get_lingbot_obs(self) -> Dict[str, np.ndarray]:
        """The 2-key uint8 dict the LingBot server consumes (same schema as the
        LIBERO client's ``_extract_obs``)."""
        assert self._last_obs is not None, "call reset() before get_lingbot_obs()"
        if not self._resolved_cam:
            self._resolve_camera_keys(self._last_obs)
        return {
            lkey: self._img(self._last_obs, src)
            for lkey, src in self._resolved_cam.items()
        }

    # -- action conversion -------------------------------------------------
    def expand_action(self, model_action_7d: np.ndarray) -> np.ndarray:
        """``[dx,dy,dz,drx,dry,drz,grip]`` (7,) -> full env action vector.

        Arm + gripper slots are filled; base / torso / mode stay 0 so the
        mobile base is stationary during manipulation."""
        cfg = self.cfg
        a = np.asarray(model_action_7d, dtype=np.float64).reshape(-1)
        if a.shape[0] < 7:
            a = np.pad(a, (0, 7 - a.shape[0]))
        adim = cfg.action_dim or int(self.env.action_dim)
        full = np.zeros(adim, dtype=np.float64)

        s, e = cfg.arm_action_slice
        e = min(e, s + 6)
        n = e - s
        full[s:e] = a[:n] * cfg.arm_action_scale

        gi = cfg.gripper_index
        if 0 <= gi < adim:
            full[gi] = a[6] * cfg.gripper_scale * cfg.gripper_sign

        if cfg.clip_to_action_spec:
            try:
                low, high = self.env.action_spec
                full = np.clip(full, np.asarray(low), np.asarray(high))
            except Exception:
                full = np.clip(full, -1.0, 1.0)
        return full

    # -- lifecycle ---------------------------------------------------------
    def _read_task_language(self) -> str:
        for getter in ("get_ep_meta",):
            fn = getattr(self.env, getter, None)
            if callable(fn):
                try:
                    meta = fn() or {}
                    lang = meta.get("lang") or meta.get("language")
                    if lang:
                        return str(lang)
                except Exception:
                    pass
        for attr in ("lang", "language"):
            v = getattr(self.env, attr, None)
            if v:
                return str(v)
        # Fall back to a readable form of the env name.
        name = self.cfg.env_name
        return "".join(" " + c.lower() if c.isupper() else c for c in name).strip()

    def reset(self) -> Dict[str, np.ndarray]:
        raw = self.env.reset()
        self._last_obs = raw
        self._resolved_cam = {}
        self._ep_steps = 0
        self._task_language = self._read_task_language()
        # warm-up no-op steps (mirrors LIBERO client's settle loop)
        for _ in range(5):
            try:
                zero = np.zeros(self.cfg.action_dim or int(self.env.action_dim))
                raw, _, _, _ = self.env.step(zero)
                self._last_obs = raw
            except Exception:
                break
        return self.get_lingbot_obs()

    def step(self, model_action_7d: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool]:
        """Apply one model action. Returns (lingbot_obs, done) where ``done`` is
        True on task success or step-cap (LIBERO-client semantics)."""
        full = self.expand_action(model_action_7d)
        raw, _, _, _ = self.env.step(full)
        self._last_obs = raw
        self._ep_steps += 1
        done = self.check_success() or (self._ep_steps >= self.cfg.max_env_steps)
        return self.get_lingbot_obs(), done

    def check_success(self) -> bool:
        for m in ("_check_success", "check_success"):
            fn = getattr(self.env, m, None)
            if callable(fn):
                try:
                    res = fn()
                    if isinstance(res, dict):
                        return bool(res.get("task", False) or all(res.values()))
                    return bool(res)
                except Exception:
                    pass
        return False

    @property
    def task_language(self) -> str:
        return self._task_language

    @property
    def ep_steps(self) -> int:
        return self._ep_steps

    def render_views(self) -> Dict[str, np.ndarray]:
        """Higher-res views for saved videos (not downsampled)."""
        return self.get_lingbot_obs()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


# Curated default task set (representative occlusion-prone Robocasa atomic
# tasks). Names verified against the robosuite 1.5.2 / robocasa 1.0.1 registry
# on the server (see probe output). Override via the client's --tasks flag
# with any name from `python -m robocasa.demos.demo_tasks`. For long-horizon
# demonstrations of CoT, use composite tasks e.g. ArrangeVegetables,
# PrepareCoffee, MakeFruitBowl, OrganizeVegetables.
DEFAULT_TASKS: Tuple[str, ...] = (
    "PickPlaceCounterToCabinet",    # occlusion: object goes inside a cabinet
    "PickPlaceCounterToMicrowave",  # occlusion + articulated appliance
    "OpenDrawer",                   # articulated, prerequisite-style
    "PickPlaceCounterToSink",       # long reach, container
)

# Known-good env names tried by probe_env.py if the requested one is absent.
PROBE_FALLBACK_ENVS: Tuple[str, ...] = (
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToMicrowave",
    "OpenDrawer",
    "OpenCabinet",
)
