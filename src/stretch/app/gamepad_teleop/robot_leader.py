#!/usr/bin/env python3
"""
robot_leader.py — Stretch SE3 gamepad teleop + data collection (ON ROBOT).

Runs ON the robot; uses stretch_body directly for silky-smooth velocity
control (matching the official stretch_gamepad_teleop daemon's pattern).
Reads dual RealSense cameras directly via pyrealsense2.

The bridge container should NOT be running while this script is active —
both want to hold the same /dev/hello-* hardware ports.

Gamepad mapping — modeled after stretch_gamepad_teleop daemon
─────────────────────────────────────────────────────────────────
  Left stick    X = base yaw       Y = base linear
  Right stick   X = arm extend     Y = lift
  LB / RB       wrist_yaw  (left = +, right = −)   ALWAYS
  D-pad         wrist pitch (up/down) + roll (L/R)
  A button      close gripper  (hold)
  B button      open  gripper  (hold)
  L2 trigger    precision mode (velocities × 0.3 while held)
  R2 trigger    fast-base mode (base × 1.5  while held)
  X button      toggle head between 'ahead' and 'tool' (look at gripper)
  Y button      save SUCCESS  (record control)
  BACK button   print info
  START button  quit
  LB+RB combo   toggle recording (start/pause)

Console commands (typed during run):
  s, start    start a new recording episode
  p, pause    stop recording but keep buffered (no save)
  y, save     save current episode as SUCCESS
  a, abandon  discard current episode buffer
  i, info     print full state snapshot
  q, quit     exit cleanly
"""

import argparse
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# stretch_body needs these at IMPORT TIME (interactive bashrc doesn't run
# under ssh non-interactive shells)
os.environ.setdefault("HELLO_FLEET_PATH", "/home/hello-robot/stretch_user")
os.environ.setdefault("HELLO_FLEET_ID",   "stretch-se3-3139")

# ── Stretch hardware ─────────────────────────────────────────────────
import stretch_body.robot as rb

# ── Cameras ──────────────────────────────────────────────────────────
import pyrealsense2 as rs

# ── Gamepad ──────────────────────────────────────────────────────────
import evdev

# ── Recorder (compat with stretch_ai dex_teleop format) ──────────────
sys.path.insert(0, "/home/hello-robot/stretch_ai/src")
from stretch.utils.data_tools.record import FileDataRecorder


# ══════════════════════════════════════════════════════════════════════
#                              CONSTANTS
# ══════════════════════════════════════════════════════════════════════
DEFAULT_FPS = 15
# Gripper position scale is in custom units, NOT [-0.3, 0.6].
# Empirically: 0 = closed, ~9 = comfortably open, up to ~70 fully open.
# Use wide soft-limits and let the Dynamixel HW limit clamp.
GRIPPER_CLOSED = -2.0
GRIPPER_OPEN   = 80.0

# Joint limits (m or rad)
LIFT_MIN, LIFT_MAX = 0.15, 1.10
ARM_MIN,  ARM_MAX  = 0.00, 0.50
WRIST_YAW_MIN,   WRIST_YAW_MAX   = -1.50, 1.50
WRIST_PITCH_MIN, WRIST_PITCH_MAX = -1.40, 0.40
WRIST_ROLL_MIN,  WRIST_ROLL_MAX  = -2.00, 2.00
HEAD_PAN_MIN,    HEAD_PAN_MAX    = -2.50, 2.50
HEAD_TILT_MIN,   HEAD_TILT_MAX   = -1.50, 0.50

# Velocity scaling at full stick deflection
MAX_BASE_V    = 0.30   # m/s
MAX_BASE_W    = 0.80   # rad/s
MAX_LIFT_V    = 0.20   # m/s
MAX_ARM_V     = 0.15   # m/s
MAX_WRIST_V   = 1.50   # rad/s
MAX_HEAD_V    = 1.50   # rad/s
MAX_GRIPPER_V = 3.00   # gripper-units/s (used only as on/off direction)
GRIPPER_MOVE_BY_PCT = 60.0   # daemon's gripper_rotate_pct: move target by this each call

# Acceleration (m/s² or rad/s²) — fed to stretch_body set_velocity
ACC_LIFT     = 0.20
ACC_ARM      = 0.20
ACC_WRIST    = 8.0
ACC_HEAD     = 8.0
ACC_GRIPPER  = 4.0

STICK_DEADZONE   = 0.10
TRIGGER_DEADZONE = 0.05

# 1×1 placeholder depth (used when --save_depth is off; avoids GB-scale
# liblzfse compression that hangs the robot CPU for minutes).
_DEPTH_STUB = np.zeros((1, 1), dtype=np.uint16)

# Camera config
CAM_RGB_W, CAM_RGB_H, CAM_RGB_FPS = 640, 480, 30
CAM_DEPTH_W, CAM_DEPTH_H, CAM_DEPTH_FPS = 640, 480, 30


# ══════════════════════════════════════════════════════════════════════
#                              HELPERS
# ══════════════════════════════════════════════════════════════════════
def deadzone(v, dz=STICK_DEADZONE):
    if abs(v) < dz:
        return 0.0
    return math.copysign((abs(v) - dz) / (1.0 - dz), v)


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(v, lo, hi):
    return max(lo, min(hi, v))


# ══════════════════════════════════════════════════════════════════════
#                       Xbox-style gamepad reader
# ══════════════════════════════════════════════════════════════════════
class XboxState:
    BTN_MAP = {
        evdev.ecodes.BTN_SOUTH:  "A",
        evdev.ecodes.BTN_EAST:   "B",
        evdev.ecodes.BTN_WEST:   "X",
        evdev.ecodes.BTN_NORTH:  "Y",
        evdev.ecodes.BTN_TL:     "LB",
        evdev.ecodes.BTN_TR:     "RB",
        evdev.ecodes.BTN_START:  "START",
        evdev.ecodes.BTN_SELECT: "BACK",
        evdev.ecodes.BTN_THUMBL: "LSTICK",
        evdev.ecodes.BTN_THUMBR: "RSTICK",
    }

    def __init__(self, device_path: Optional[str] = None):
        self.dev = self._auto_detect() if device_path is None \
                   else evdev.InputDevice(device_path)
        print(f"[gamepad] using {self.dev.name!r} at {self.dev.path}")

        self._abs_info = {}
        for code, info in self.dev.capabilities().get(evdev.ecodes.EV_ABS, []):
            self._abs_info[code] = info

        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        self.lt = 0.0
        self.rt = 0.0
        self.dpad_x = 0
        self.dpad_y = 0
        self.btn = {k: 0 for k in
                    ("A","B","X","Y","LB","RB","START","BACK","LSTICK","RSTICK")}
        self._prev_btn = dict(self.btn)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _auto_detect():
        devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
        keywords = ("xbox","x-box","gamepad","controller","joystick",
                    "playstation","ds4","dualshock","dualsense","8bitdo",
                    "stadia"," pad")
        excludes = ("touchscreen","touchpad","passthrough","trackpad")
        prio = [
            d for d in devs
            if any(k in d.name.lower() for k in keywords)
            and not any(x in d.name.lower() for x in excludes)
        ]
        if not prio:
            prio = [
                d for d in devs
                if evdev.ecodes.EV_ABS in d.capabilities()
                and any(c == evdev.ecodes.ABS_X for c, _ in
                        d.capabilities()[evdev.ecodes.EV_ABS])
                and not any(x in d.name.lower() for x in excludes)
            ]
        if not prio:
            raise RuntimeError("No gamepad detected.")
        return prio[0]

    def _norm(self, code, raw):
        info = self._abs_info.get(code)
        if not info:
            return 0.0
        if code in (evdev.ecodes.ABS_Z, evdev.ecodes.ABS_RZ):
            return raw / info.max if info.max > 0 else 0.0
        mid = 0.5 * (info.min + info.max)
        half = 0.5 * (info.max - info.min)
        return (raw - mid) / half if half > 0 else 0.0

    def _loop(self):
        while not self._stop.is_set():
            try:
                ev = self.dev.read_one()
            except OSError:
                time.sleep(0.1)
                continue
            if ev is None:
                time.sleep(0.001)
                continue
            if ev.type == evdev.ecodes.EV_ABS:
                v = self._norm(ev.code, ev.value)
                if   ev.code == evdev.ecodes.ABS_X:    self.left_x  = v
                elif ev.code == evdev.ecodes.ABS_Y:    self.left_y  = -v
                elif ev.code == evdev.ecodes.ABS_RX:   self.right_x = v
                elif ev.code == evdev.ecodes.ABS_RY:   self.right_y = -v
                elif ev.code == evdev.ecodes.ABS_Z:    self.lt = v
                elif ev.code == evdev.ecodes.ABS_RZ:   self.rt = v
                elif ev.code == evdev.ecodes.ABS_HAT0X:
                    self.dpad_x = 1 if ev.value > 0 else (-1 if ev.value < 0 else 0)
                elif ev.code == evdev.ecodes.ABS_HAT0Y:
                    self.dpad_y = -1 if ev.value > 0 else (1 if ev.value < 0 else 0)
            elif ev.type == evdev.ecodes.EV_KEY:
                name = self.BTN_MAP.get(ev.code)
                if name is not None:
                    self.btn[name] = ev.value

    def edge_pressed(self, name):
        was = self._prev_btn.get(name, 0)
        now = self.btn.get(name, 0)
        edge = (was == 0 and now == 1)
        self._prev_btn[name] = now
        return edge

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════
#                  Dual RealSense cameras (head + ee)
# ══════════════════════════════════════════════════════════════════════
class DualRealsense:
    def __init__(self):
        # First: hardware_reset every camera to recover from any prior
        # frozen-pipe state (RealSense USB drivers occasionally lock up
        # if the previous owner didn't shut down cleanly).
        ctx = rs.context()
        for d in ctx.devices:
            try:
                print(f"[cam ] hardware_reset {d.get_info(rs.camera_info.name)} "
                      f"sn={d.get_info(rs.camera_info.serial_number)}")
                d.hardware_reset()
            except Exception as e:
                print(f"[cam ] reset failed: {e}")

        # Wait for cameras to re-enumerate after reset
        print("[cam ] waiting 5s for cameras to come back ...")
        time.sleep(5.0)

        # Re-enumerate
        ctx = rs.context()
        self.head_serial = None
        self.ee_serial   = None
        for d in ctx.devices:
            name = d.get_info(rs.camera_info.name)
            sn = d.get_info(rs.camera_info.serial_number)
            print(f"[cam ] found: {name}  sn={sn}")
            if "D435" in name:
                self.head_serial = sn
            elif "D405" in name:
                self.ee_serial = sn

        if self.head_serial is None:
            print("[cam ] WARNING D435 (head) not found")
        if self.ee_serial is None:
            print("[cam ] WARNING D405 (gripper) not found")

        # Both cameras at 15 fps — matches our 15 Hz recording loop, so no
        # wasted frames / bandwidth.  Head keeps depth (USB 3.0 has room),
        # EE drops depth (USB 2.0 + we record RGB only by default).
        self.head_pipe = self._make_pipeline(self.head_serial, fps=15, want_depth=True) \
                         if self.head_serial else None
        self.ee_pipe   = self._make_pipeline(self.ee_serial,   fps=15, want_depth=False) \
                         if self.ee_serial   else None

        # Cache of last-good frame per camera; consumers always get
        # a non-None RGB/depth (stale-but-valid > black).
        self._last_head_color = None
        self._last_head_depth = None
        self._last_ee_color   = None
        self._last_ee_depth   = None

        # Prime cache by blocking-wait for first frame from each camera.
        # 10s timeout — D405 on USB 2.0 can be slow to start streaming.
        for label, pipe_attr in (("head", "head_pipe"), ("ee", "ee_pipe")):
            pipe = getattr(self, pipe_attr)
            if pipe is None:
                continue
            primed = False
            for attempt in range(2):  # retry once with lighter config
                try:
                    f = pipe.wait_for_frames(10000)
                    cf = f.get_color_frame()
                    if cf:
                        color = np.asanyarray(cf.get_data()).copy()
                        depth = None
                        df = f.get_depth_frame()
                        if df:
                            depth = np.asanyarray(df.get_data()).copy()
                        if label == "head":
                            self._last_head_color = color
                            self._last_head_depth = depth
                        else:
                            self._last_ee_color = color
                            self._last_ee_depth = depth
                        print(f"[cam ] {label} primed: shape={color.shape} "
                              f"mean={color.mean():.0f}"
                              + ("" if depth is None else "  +depth"))
                        primed = True
                        break
                except Exception as e:
                    print(f"[cam ] {label} wait_for_frames attempt {attempt+1} failed: {e}")
                    # retry: stop, create lighter pipeline (RGB-only @ 15fps), re-prime
                    if attempt == 0:
                        try: pipe.stop()
                        except: pass
                        sn = self.head_serial if label == "head" else self.ee_serial
                        pipe = self._make_pipeline(sn, fps=15, want_depth=False)
                        if pipe is None:
                            break
                        setattr(self, pipe_attr, pipe)
                        time.sleep(1.0)
            if not primed:
                print(f"[cam ] {label} FAILED to prime; will run with no {label} frames")

        # Brief poll-based warm-up after first frame, so subsequent polls
        # in the main loop have hot-path coverage.
        for _ in range(10):
            self.read()
            time.sleep(0.03)

    @staticmethod
    def _make_pipeline(serial, *, want_depth=True, fps=30):
        """Try a few configs, falling back to lighter ones if start fails."""
        configs = [
            # (rgb_size, depth_size, fps, with_depth)
            ((CAM_RGB_W, CAM_RGB_H), (CAM_DEPTH_W, CAM_DEPTH_H), fps, want_depth),
            ((CAM_RGB_W, CAM_RGB_H), (CAM_DEPTH_W, CAM_DEPTH_H), 15,  want_depth),
            ((CAM_RGB_W, CAM_RGB_H), None,                       15,  False),  # RGB-only
            ((424, 240),             None,                       15,  False),  # smaller RGB
        ]
        for (cw, ch), depth_dim, f, wd in configs:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, f)
            if wd and depth_dim:
                dw, dh = depth_dim
                cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, f)
            try:
                pipe.start(cfg)
                d_str = f"+depth({depth_dim[0]}x{depth_dim[1]})" if (wd and depth_dim) else ""
                print(f"[cam ] sn={serial} pipeline up: rgb={cw}x{ch}{d_str} @{f}fps")
                return pipe
            except Exception as e:
                print(f"[cam ] sn={serial} cfg rgb={cw}x{ch} fps={f} depth={wd} failed: {e}")
        return None

    def _grab(self, pipe):
        """Return (color, depth) if a fresh frame is ready.
        Depth is None if the pipeline doesn't have a depth stream enabled.
        """
        if pipe is None:
            return None, None
        try:
            frames = pipe.poll_for_frames()
            if not frames:
                return None, None
            cf = frames.get_color_frame()
            if not cf:
                return None, None
            color = np.asanyarray(cf.get_data()).copy()
            depth = None
            df = frames.get_depth_frame()
            if df:
                depth = np.asanyarray(df.get_data()).copy()
            return color, depth
        except Exception:
            return None, None

    def read(self):
        """Return latest (head_rgb, head_depth, ee_rgb, ee_depth).

        If a camera doesn't have a fresh frame this cycle, returns the
        cached last-good frame instead of None. This prevents black frames
        in the recording when the camera and control loop fall out of phase.
        """
        hc, hd = self._grab(self.head_pipe)
        if hc is not None:
            self._last_head_color = hc
            self._last_head_depth = hd
        ec, ed = self._grab(self.ee_pipe)
        if ec is not None:
            self._last_ee_color = ec
            self._last_ee_depth = ed
        return (self._last_head_color, self._last_head_depth,
                self._last_ee_color,   self._last_ee_depth)

    def stop(self):
        for p in (self.head_pipe, self.ee_pipe):
            if p:
                try: p.stop()
                except: pass


# ══════════════════════════════════════════════════════════════════════
#                       Stretch motor controller
# ══════════════════════════════════════════════════════════════════════
class StretchController:
    """Thin wrapper around stretch_body.Robot.  Per-joint velocity setting
    in the same pattern as stretch_gamepad_teleop daemon (= silky smooth).
    """

    def __init__(self):
        self.robot = rb.Robot()
        ok = self.robot.startup()
        if not ok:
            raise RuntimeError("stretch_body Robot.startup() returned False")

        if not self.robot.is_calibrated():
            raise RuntimeError(
                "Robot is NOT calibrated/homed.  Run stretch_robot_home.py first.")
        self.has_dexwrist = True  # SE3 has dex wrist by default

        # Cache gripper motion params (max vel + accel from robot params)
        gp_params = self.robot.end_of_arm.motors["stretch_gripper"].params
        self._gripper_max_vel = gp_params["motion"]["max"]["vel"]
        self._gripper_max_acc = gp_params["motion"]["max"]["accel"]
        print(f"[motor] stretch_body OK; gripper vel={self._gripper_max_vel} "
              f"acc={self._gripper_max_acc}")

    # ─ State read ─────────────────────────────────────────────────────
    def read_state(self):
        b   = self.robot.base.status
        eoa = self.robot.end_of_arm.motors
        h   = self.robot.head.motors
        return {
            # base — odom-frame (no SLAM)
            "base_x":      b["x"],
            "base_y":      b["y"],
            "base_theta":  b["theta"],
            "v_forward":   b.get("x_vel", 0.0),
            "v_yaw":       b.get("theta_vel", 0.0),
            # arm
            "lift":        self.robot.lift.status["pos"],
            "arm":         self.robot.arm.status["pos"],
            # wrist
            "wrist_yaw":   eoa["wrist_yaw"].status["pos"],
            "wrist_pitch": eoa["wrist_pitch"].status["pos"],
            "wrist_roll":  eoa["wrist_roll"].status["pos"],
            "gripper":     eoa["stretch_gripper"].status["pos"],
            # head
            "head_pan":    h["head_pan"].status["pos"],
            "head_tilt":   h["head_tilt"].status["pos"],
        }

    # ─ Velocity command (per joint) ───────────────────────────────────
    def send_velocities(self, v, skip_head: bool = False):
        """Send all per-joint velocities, then push_command once.

        Gripper uses move_by() like the daemon (set_velocity gets clamped
        early by Dynamixel internal target-update behavior).

        skip_head: when True, don't send head set_velocity (used while
        a head.pose() position-target move is settling, so we don't
        override it with v=0).
        """
        # Hello-Motor lift / arm
        self.robot.lift.set_velocity(v.get("lift", 0.0), a_m=ACC_LIFT)
        self.robot.arm.set_velocity (v.get("arm",  0.0), a_m=ACC_ARM)

        # Base — combined v+w
        self.robot.base.set_velocity(v.get("base_v", 0.0),
                                     v.get("base_w", 0.0))

        # Wrist Dynamixels via set_velocity
        for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            self.robot.end_of_arm.set_velocity(j, v.get(j, 0.0), a_r=ACC_WRIST)

        # Head — skipped during pose move so we don't fight the position target
        if not skip_head:
            for j in ("head_pan", "head_tilt"):
                self.robot.head.set_velocity(j, v.get(j, 0.0), a_r=ACC_HEAD)

        # Gripper: daemon-style move_by (60 per call, max vel/accel)
        gv = v.get("gripper", 0.0)
        if gv != 0.0:
            grip = self.robot.end_of_arm.get_joint("stretch_gripper")
            pct = math.copysign(GRIPPER_MOVE_BY_PCT, gv)
            try:
                grip.move_by(pct,
                             self._gripper_max_vel,
                             self._gripper_max_acc)
            except Exception as e:
                print(f"[motor] gripper move_by failed: {e}")

        # Flush queued Hello-Motor commands (lift/arm/base only)
        self.robot.push_command()

    def stop_all(self):
        self.send_velocities({})

    # Custom head poses (override stretch_body defaults)
    #   'ahead' has tilt=-0.4 (~-23°) so the floor is visible while
    #     driving / approaching objects.  stretch_body default is (0,0)
    #     which only sees walls.
    #   'tool'  matches stretch_body default — looks at the gripper.
    _HEAD_POSES = {
        "ahead": (0.0, -0.40),                # (pan, tilt) rad
        "tool":  (-math.pi / 2, -math.pi / 4),  # -90, -45 deg
    }

    def head_pose(self, name: str):
        """Move head to a named pose. Returns approx motion ETA (seconds)."""
        if name not in self._HEAD_POSES:
            print(f"[motor] unknown head pose {name!r}")
            return 0.0
        pan, tilt = self._HEAD_POSES[name]
        try:
            self.robot.head.move_to("head_pan",  pan)
            self.robot.head.move_to("head_tilt", tilt)
            return 1.5
        except Exception as e:
            print(f"[motor] head_pose({name!r}) failed: {e}")
            return 0.0

    def shutdown(self):
        try:
            self.stop_all()
        except Exception:
            pass
        self.robot.stop()


# ══════════════════════════════════════════════════════════════════════
#                  Gamepad → per-joint velocity mapping
#                  (modeled after stretch_gamepad_teleop daemon)
# ══════════════════════════════════════════════════════════════════════
def map_gamepad(gp: XboxState, state: dict) -> dict:
    """Daemon-style mapping. D-pad always controls wrist pitch+roll."""
    # Modifier scaling
    precision  = gp.lt > 0.7      # L2 held → fine motion
    fast_base  = gp.rt > 0.7      # R2 held → faster base
    p_scale = 0.30 if precision else 1.0
    base_boost = 1.5 if fast_base else 1.0

    # ─ Base (left stick) ─────────────────────────────────────────────
    v_fwd = deadzone(gp.left_y) * MAX_BASE_V * base_boost * p_scale
    v_yaw = -deadzone(gp.left_x) * MAX_BASE_W * base_boost * p_scale

    # ─ Arm + Lift (right stick) ──────────────────────────────────────
    rx = deadzone(gp.right_x)
    ry = deadzone(gp.right_y)
    v_arm  = rx * MAX_ARM_V  * p_scale
    v_lift = ry * MAX_LIFT_V * p_scale

    # ─ Wrist yaw (shoulder buttons) ──────────────────────────────────
    # LB pressed = +1, RB pressed = -1 (matches daemon)
    yaw_dir = (1 if gp.btn["LB"] else 0) - (1 if gp.btn["RB"] else 0)
    v_wy = yaw_dir * MAX_WRIST_V * p_scale

    # ─ D-pad → wrist pitch + roll (always) ──────────────────────────
    v_wp = gp.dpad_y * MAX_WRIST_V * p_scale          # up=+pitch
    v_wr = gp.dpad_x * MAX_WRIST_V * p_scale          # right=+roll
    v_hp = v_ht = 0.0   # head not driven by sticks; use X-button presets

    # ─ Gripper (A=close, B=open) ─────────────────────────────────────
    grip_dir = (1 if gp.btn["B"] else 0) - (1 if gp.btn["A"] else 0)
    v_grip = grip_dir * MAX_GRIPPER_V * p_scale

    # ─ Soft joint-limit clamps ───────────────────────────────────────
    def softlimit(cur, lo, hi, v):
        if v > 0 and cur >= hi: return 0.0
        if v < 0 and cur <= lo: return 0.0
        return v

    v_lift = softlimit(state["lift"],        LIFT_MIN, LIFT_MAX, v_lift)
    v_arm  = softlimit(state["arm"],         ARM_MIN,  ARM_MAX,  v_arm)
    v_wy   = softlimit(state["wrist_yaw"],   WRIST_YAW_MIN, WRIST_YAW_MAX, v_wy)
    v_wp   = softlimit(state["wrist_pitch"], WRIST_PITCH_MIN, WRIST_PITCH_MAX, v_wp)
    v_wr   = softlimit(state["wrist_roll"],  WRIST_ROLL_MIN, WRIST_ROLL_MAX, v_wr)
    v_grip = softlimit(state["gripper"],     GRIPPER_CLOSED, GRIPPER_OPEN, v_grip)
    v_hp   = softlimit(state["head_pan"],    HEAD_PAN_MIN, HEAD_PAN_MAX, v_hp)
    v_ht   = softlimit(state["head_tilt"],   HEAD_TILT_MIN, HEAD_TILT_MAX, v_ht)

    return {
        "lift":         v_lift,
        "arm":          v_arm,
        "wrist_yaw":    v_wy,
        "wrist_pitch":  v_wp,
        "wrist_roll":   v_wr,
        "gripper":      v_grip,
        "head_pan":     v_hp,
        "head_tilt":    v_ht,
        "base_v":       v_fwd,
        "base_w":       v_yaw,
    }


# ══════════════════════════════════════════════════════════════════════
#                       Console input thread
# ══════════════════════════════════════════════════════════════════════
class ConsoleInput:
    """Background stdin reader; main loop polls cmd queue."""

    def __init__(self):
        self.q = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    self.q.put("quit")
                    return
                cmd = line.strip().lower()
                if cmd:
                    self.q.put(cmd)
            except Exception:
                return

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


# ══════════════════════════════════════════════════════════════════════
#                              Main loop
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Robot-side gamepad teleop + recorder")
    ap.add_argument("--task",     default="default_task")
    ap.add_argument("--user",     default="default_user")
    ap.add_argument("--env",      default="default_env")
    ap.add_argument("--data_dir", default="/home/hello-robot/stretch_data")
    ap.add_argument("--fps",      type=int, default=DEFAULT_FPS)
    ap.add_argument("--save_depth", action="store_true",
                    help="store full-resolution depth (slow on long episodes; "
                         "single-threaded liblzfse compression of GB-scale data "
                         "can take minutes on the robot CPU). Default off.")
    ap.add_argument("--image_size", type=int, default=0,
                    help="if >0, resize RGB to image_size x image_size before "
                         "recording. Recommended: 224 (matches pi0.5 / OpenVLA "
                         "PaliGemma SigLIP input directly). 128 is what "
                         "RoboCasa stores but model still resizes to 224 at "
                         "training (lossy). 0 = keep camera native 640x480 "
                         "(max flexibility, ~6.7x bigger files).")
    ap.add_argument("--head_rotate", type=int, default=90,
                    choices=[0, 90, 180, 270],
                    help="rotate head image clockwise by N degrees before "
                         "saving. Stretch's D435if is mounted sideways so the "
                         "raw image is rotated; default 90 corrects to "
                         "upright.")
    ap.add_argument("--video_crf", type=int, default=15,
                    help="ffmpeg h264 CRF (lower = better quality, bigger "
                         "file). 12=near-original-quality, 18=visually "
                         "lossless, 23=ffmpeg default, 30=FileDataRecorder "
                         "default (too lossy). Our default 15 = very crisp.")
    args = ap.parse_args()

    dt = 1.0 / args.fps

    print("──────────────────────────────────────────────────────────────")
    print("  Stretch SE3 robot-side gamepad leader + data recorder")
    print("  task =", args.task, "user =", args.user, "env =", args.env)
    print("  data_dir =", args.data_dir, "fps =", args.fps)
    print("──────────────────────────────────────────────────────────────")
    print()
    print("Console commands:")
    print("  s, start    start recording      i, info    print state")
    print("  p, pause    pause (no save)      q, quit    exit")
    print("  y, save     save SUCCESS")
    print("  a, abandon  discard buffer")
    print()

    # ── Init hardware ────────────────────────────────────────────────
    motor = StretchController()
    gp    = XboxState()
    cams  = DualRealsense()
    recorder = FileDataRecorder(
        datadir=args.data_dir,
        task=args.task, user=args.user, env=args.env,
        save_images=False,
        metadata={"task": args.task, "user": args.user, "env": args.env,
                  "leader": "robot_gamepad", "fps": args.fps,
                  "video_crf": args.video_crf,
                  "head_rotate_deg": args.head_rotate,
                  "image_size": args.image_size or "native"},
        fps=args.fps,
    )

    # Monkey-patch the recorder's video-encoding step to use OUR crf
    # (FileDataRecorder hardcodes crf=30 which is too aggressive for VLA).
    _orig_proc_rgb = recorder.process_rgb_to_video

    def _proc_rgb_with_crf(episode_dir, head=False, *args_, **kwargs):
        # Re-implement just enough to substitute crf
        import subprocess
        if head:
            from stretch.utils.data_tools.record import (
                HEAD_RGB_FOLDER_NAME as RGB_FOLDER,
                HEAD_RGB_VIDEO_H264_NAME as VIDEO_NAME,
            )
        else:
            from stretch.utils.data_tools.record import (
                RGB_FOLDER_NAME as RGB_FOLDER,
                RGB_VIDEO_H264_NAME as VIDEO_NAME,
            )
        rgb_dir = episode_dir / RGB_FOLDER
        video_path = episode_dir / VIDEO_NAME
        try:
            sample = next(rgb_dir.glob("*.png"))
        except StopIteration:
            return
        fmt = "%06d.png" if len(sample.stem) == 6 else "%04d.png"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(rgb_dir / fmt),
            "-c:v", "libx264", "-crf", str(args.video_crf),
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            str(video_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"[REC ] encoded {video_path.name} (crf={args.video_crf})")

    recorder.process_rgb_to_video = _proc_rgb_with_crf
    console = ConsoleInput()

    recording  = False
    episode_idx = 0
    next_t = time.time()
    frame_i = 0
    last_print = time.time()

    # Head pose toggle state (X button cycles between these two)
    head_pose_state = "ahead"        # 'ahead' or 'tool'
    head_lock_until = 0.0             # while time.time() < this, skip head set_velocity

    print("READY.  Type 's' (start recording) or use gamepad to teleop.")
    print("    button cheat-sheet:")
    print("      A=close-grip   B=open-grip   X=head pose (ahead↔tool)")
    print("      Y=save success   BACK=info   START=quit")
    print("      LB/RB=wrist_yaw   L2=precision   R2=fast-base")
    print("      D-pad=wrist pitch/roll   LB+RB=toggle recording")
    print()

    try:
        while True:
            t_loop = time.time()

            # 1) Read full state (encoders + odom)
            state = motor.read_state()

            # 2) Read gamepad → per-joint velocities
            vels = map_gamepad(gp, state)

            # 3) Send to motors (skip head set_velocity during pose move)
            skip_head = time.time() < head_lock_until
            motor.send_velocities(vels, skip_head=skip_head)

            # 4) Read cameras (always; cheap if no record)
            head_rgb, head_depth, ee_rgb, ee_depth = cams.read()

            # 4a) Rotate head camera (D435if is sideways-mounted on Stretch)
            if args.head_rotate and head_rgb is not None:
                k = args.head_rotate // 90      # CW: 1, 2, or 3
                head_rgb = np.rot90(head_rgb, k=-k).copy()
                if head_depth is not None:
                    head_depth = np.rot90(head_depth, k=-k).copy()

            # 4b) Optional pre-resize for direct VLA target size
            if args.image_size > 0:
                import cv2 as _cv
                S = args.image_size
                if head_rgb is not None:
                    head_rgb = _cv.resize(head_rgb, (S, S), interpolation=_cv.INTER_AREA)
                if ee_rgb is not None:
                    ee_rgb = _cv.resize(ee_rgb, (S, S), interpolation=_cv.INTER_AREA)
                if args.save_depth:
                    if head_depth is not None:
                        head_depth = _cv.resize(head_depth, (S, S),
                                                interpolation=_cv.INTER_NEAREST)
                    if ee_depth is not None:
                        ee_depth = _cv.resize(ee_depth, (S, S),
                                              interpolation=_cv.INTER_NEAREST)

            # 5) Compute action targets (= state + v×dt) for recording
            tgt = {
                "joint_lift":         clip(state["lift"]        + vels["lift"]        * dt, LIFT_MIN, LIFT_MAX),
                "joint_arm_l0":       clip(state["arm"]         + vels["arm"]         * dt, ARM_MIN, ARM_MAX),
                "joint_wrist_yaw":    clip(state["wrist_yaw"]   + vels["wrist_yaw"]   * dt, WRIST_YAW_MIN, WRIST_YAW_MAX),
                "joint_wrist_pitch":  clip(state["wrist_pitch"] + vels["wrist_pitch"] * dt, WRIST_PITCH_MIN, WRIST_PITCH_MAX),
                "joint_wrist_roll":   clip(state["wrist_roll"]  + vels["wrist_roll"]  * dt, WRIST_ROLL_MIN, WRIST_ROLL_MAX),
                "stretch_gripper":    clip(state["gripper"]     + vels["gripper"]     * dt, GRIPPER_CLOSED, GRIPPER_OPEN),
                "joint_head_pan":     clip(state["head_pan"]    + vels["head_pan"]    * dt, HEAD_PAN_MIN, HEAD_PAN_MAX),
                "joint_head_tilt":    clip(state["head_tilt"]   + vels["head_tilt"]   * dt, HEAD_TILT_MIN, HEAD_TILT_MAX),
                # base — robot-local delta (per-frame)
                "delta_s_robot":      vels["base_v"] * dt,
                "delta_theta":        vels["base_w"] * dt,
                # raw velocity commands kept for completeness
                "v_lift":             vels["lift"],
                "v_arm":              vels["arm"],
                "v_wrist_yaw":        vels["wrist_yaw"],
                "v_wrist_pitch":      vels["wrist_pitch"],
                "v_wrist_roll":       vels["wrist_roll"],
                "v_gripper":          vels["gripper"],
                "v_head_pan":         vels["head_pan"],
                "v_head_tilt":        vels["head_tilt"],
                "v_base_forward":     vels["base_v"],
                "v_base_yaw":         vels["base_w"],
                "leader":             "robot_gamepad",
            }

            # 6) Record frame
            if recording:
                obs_dict = {
                    "base_x":          state["base_x"],
                    "base_y":          state["base_y"],
                    "base_theta":      state["base_theta"],
                    "lift":            state["lift"],
                    "arm":             state["arm"],
                    "gripper":         state["gripper"],
                    "wrist_roll":      state["wrist_roll"],
                    "wrist_pitch":     state["wrist_pitch"],
                    "wrist_yaw":       state["wrist_yaw"],
                    "head_pan":        state["head_pan"],
                    "head_tilt":       state["head_tilt"],
                    "base_v_forward":  state["v_forward"],
                    "base_v_yaw":      state["v_yaw"],
                }
                # FileDataRecorder requires non-None images; substitute zeros if missing
                _S = args.image_size if args.image_size > 0 else None
                _h = _S or CAM_RGB_H
                _w = _S or CAM_RGB_W
                _zh = np.zeros((_h, _w, 3), dtype=np.uint8)
                if args.save_depth:
                    _zd = np.zeros((_h, _w), dtype=np.uint16)
                    rec_ee_depth   = ee_depth   if ee_depth   is not None else _zd
                    rec_head_depth = head_depth if head_depth is not None else _zd
                else:
                    # Tiny 1x1 placeholder — np.stack works, liblzfse is instant.
                    rec_ee_depth   = _DEPTH_STUB
                    rec_head_depth = _DEPTH_STUB
                try:
                    recorder.add(
                        ee_rgb       = ee_rgb       if ee_rgb       is not None else _zh,
                        ee_depth     = rec_ee_depth,
                        xyz          = np.zeros(3, dtype=np.float32),
                        quaternion   = np.array([0,0,0,1], dtype=np.float32),
                        gripper      = float(state["gripper"]),
                        ee_pos       = np.zeros(3, dtype=np.float32),
                        ee_rot       = np.eye(3, dtype=np.float32),
                        observations = obs_dict,
                        actions      = tgt,
                        head_rgb     = head_rgb     if head_rgb     is not None else _zh,
                        head_depth   = rec_head_depth,
                    )
                except Exception as e:
                    print(f"[REC ] add() failed: {e}")

            # 7) Handle gamepad button events (A/B reserved for gripper)
            if gp.edge_pressed("START"):
                console.q.put("quit")
            if gp.edge_pressed("Y"):           # save SUCCESS (recording only)
                console.q.put("save")
            if gp.edge_pressed("BACK"):        # diagnostic dump
                console.q.put("info")
            if gp.edge_pressed("X"):           # toggle head pose preset
                head_pose_state = "tool" if head_pose_state == "ahead" else "ahead"
                eta = motor.head_pose(head_pose_state)
                head_lock_until = time.time() + eta
                tag = "tool (look at gripper)" if head_pose_state == "tool" else "ahead (forward)"
                print(f"\n[head] → {tag}\n")
            # Combo: LB+RB held simultaneously → toggle recording
            if (gp.btn["LB"] and gp.btn["RB"]
                and (gp.edge_pressed("LB") or gp.edge_pressed("RB"))):
                console.q.put("pause" if recording else "start")

            # 8) Handle console commands
            cmd = console.get_nowait()
            if cmd is not None:
                if cmd in ("s", "start"):
                    if not recording:
                        episode_idx += 1
                        recording = True
                        print(f"\n[REC ] >>> START episode #{episode_idx}\n")
                elif cmd in ("p", "pause"):
                    if recording:
                        recording = False
                        print("\n[REC ] <<< PAUSED (use 'save' or 'abandon')\n")
                elif cmd in ("y", "save"):
                    if recorder.step > 0:
                        n = recorder.step
                        recording = False
                        recorder.write(success=True)
                        # Find the just-written episode dir and stitch a
                        # head+gripper side-by-side video.
                        ep_dir = _latest_episode_dir(recorder.task_dir)
                        if ep_dir is not None:
                            _make_stitched_video(ep_dir)
                        print(f"\n[REC ] ✓ saved SUCCESS  ({n} frames)\n")
                    else:
                        print("[REC ] (save) no frames buffered")
                elif cmd in ("a", "abandon"):
                    if recorder.step > 0:
                        n = recorder.step
                        recorder.reset()
                        print(f"\n[REC ] ✗ abandoned  ({n} frames discarded)\n")
                    recording = False
                elif cmd in ("i", "info"):
                    _print_full(state, vels, tgt, recording, recorder.step)
                elif cmd in ("q", "quit", "exit"):
                    print("\n[exit] quit requested.")
                    break
                elif cmd in ("h", "help", "?"):
                    print(__doc__.split("Console commands:")[1].split("Gamepad")[0])
                else:
                    print(f"[?   ] unknown cmd: {cmd!r}")

            # 9) Periodic 1-second state line
            if time.time() - last_print >= 1.0:
                rec_tag = f"REC #{episode_idx}/{recorder.step}f" if recording else "----"
                print(f"[{rec_tag}] base=({state['base_x']:+.2f},{state['base_y']:+.2f},"
                      f"{state['base_theta']:+.2f}) v=({state['v_forward']:+.2f},"
                      f"{state['v_yaw']:+.2f}) "
                      f"lift={state['lift']:.2f} arm={state['arm']:.2f} "
                      f"yaw={state['wrist_yaw']:+.2f} grip={state['gripper']:+.2f} "
                      f"head=({state['head_pan']:+.2f},{state['head_tilt']:+.2f})")
                last_print = time.time()

            # 10) Pacing
            frame_i += 1
            next_t += dt
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.time()

    finally:
        print("\n[exit] shutting down ...")
        try: motor.stop_all()
        except: pass
        gp.stop()
        cams.stop()
        try: motor.shutdown()
        except: pass
        print("[exit] done.")


def _latest_episode_dir(task_dir):
    """Return the most recently created subdir under task_dir."""
    try:
        subs = [p for p in task_dir.iterdir() if p.is_dir()]
        if not subs:
            return None
        return max(subs, key=lambda p: p.stat().st_mtime)
    except Exception as e:
        print(f"[stitch] cannot find episode dir: {e}")
        return None


def _make_stitched_video(episode_dir):
    """Run ffmpeg to combine head + gripper videos side-by-side.

    Produces <episode_dir>/stitched_head_ee.mp4   (h264, head | gripper).
    """
    import subprocess
    head_mp4    = episode_dir / "head_compressed_video_h264.mp4"
    gripper_mp4 = episode_dir / "gripper_compressed_video_h264.mp4"
    out_mp4     = episode_dir / "stitched_head_ee.mp4"
    if not head_mp4.exists() or not gripper_mp4.exists():
        print(f"[stitch] missing input videos in {episode_dir}")
        return
    cmd = [
        "ffmpeg", "-y",
        "-i", str(head_mp4),
        "-i", str(gripper_mp4),
        "-filter_complex",
        # Normalize both to 480 height (preserves aspect), then hstack.
        # Required because head is rotated (480x640) and ee is (480x640
        # too, or possibly different shapes).
        "[0:v]scale=-2:480[v0];[1:v]scale=-2:480[v1];[v0][v1]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "28",
        "-loglevel", "error",
        str(out_mp4),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0:
            print(f"[stitch] ✓ {out_mp4.name}")
        else:
            print(f"[stitch] ffmpeg failed (rc={r.returncode}): "
                  f"{r.stderr.decode()[:200]}")
    except Exception as e:
        print(f"[stitch] error: {e}")


def _print_full(state, vels, tgt, recording, n_frames):
    print("\n══ FULL STATE ════════════════════════════════════════")
    print(f"  recording = {recording}   buffered = {n_frames} frames")
    print(f"  base    = ({state['base_x']:+.3f}, {state['base_y']:+.3f}, "
          f"{state['base_theta']:+.3f}) rad")
    print(f"  base_v  = forward {state['v_forward']:+.3f} m/s, yaw {state['v_yaw']:+.3f} rad/s")
    print(f"  lift    = {state['lift']:.3f} m")
    print(f"  arm     = {state['arm']:.3f} m")
    print(f"  wrist   = yaw {state['wrist_yaw']:+.3f}  pitch {state['wrist_pitch']:+.3f}  "
          f"roll {state['wrist_roll']:+.3f}")
    print(f"  gripper = {state['gripper']:+.3f}  ([-0.3 close, +0.6 open])")
    print(f"  head    = pan {state['head_pan']:+.3f}, tilt {state['head_tilt']:+.3f}")
    print("\n  cmd v   = " + ", ".join(f"{k}={v:+.2f}" for k, v in vels.items() if abs(v) > 1e-4))
    print()


if __name__ == "__main__":
    main()
