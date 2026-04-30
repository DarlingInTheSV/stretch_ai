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
  D-pad         wrist pitch + roll   (DEFAULT)
                  ↕  toggle with X button  ↕
                head pan + tilt
  A button      close gripper  (hold)
  B button      open  gripper  (hold)
  L2 trigger    precision mode (velocities × 0.3 while held)
  R2 trigger    fast-base mode (base × 1.5  while held)
  X button      TOGGLE D-pad function (wrist ↔ head)
  Y button      save SUCCESS  (record control)
  BACK button   abandon current recording / print state
  START button  quit

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

        self.head_pipe = self._make_pipeline(self.head_serial) if self.head_serial else None
        self.ee_pipe   = self._make_pipeline(self.ee_serial)   if self.ee_serial   else None

        # Warm up: drop first few frames to stabilize exposure
        for _ in range(5):
            self.read()
            time.sleep(0.05)

    @staticmethod
    def _make_pipeline(serial):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, CAM_RGB_W, CAM_RGB_H, rs.format.bgr8, CAM_RGB_FPS)
        cfg.enable_stream(rs.stream.depth, CAM_DEPTH_W, CAM_DEPTH_H, rs.format.z16, CAM_DEPTH_FPS)
        try:
            pipe.start(cfg)
        except Exception as e:
            print(f"[cam ] pipeline start failed for serial {serial}: {e}")
            return None
        return pipe

    def _grab(self, pipe):
        if pipe is None:
            return None, None
        try:
            frames = pipe.poll_for_frames()
            if not frames:
                return None, None
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                return None, None
            color = np.asanyarray(cf.get_data()).copy()
            depth = np.asanyarray(df.get_data()).copy()
            return color, depth
        except Exception:
            return None, None

    def read(self):
        head_color, head_depth = self._grab(self.head_pipe)
        ee_color,   ee_depth   = self._grab(self.ee_pipe)
        return head_color, head_depth, ee_color, ee_depth

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
    def send_velocities(self, v):
        """Send all per-joint velocities, then push_command once.

        Gripper uses move_by() like the daemon (set_velocity gets clamped
        early by Dynamixel internal target-update behavior).
        """
        # Hello-Motor lift / arm
        self.robot.lift.set_velocity(v.get("lift", 0.0), a_m=ACC_LIFT)
        self.robot.arm.set_velocity (v.get("arm",  0.0), a_m=ACC_ARM)

        # Base — combined v+w
        self.robot.base.set_velocity(v.get("base_v", 0.0),
                                     v.get("base_w", 0.0))

        # Dynamixel chain — set_velocity for wrist + head
        for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            self.robot.end_of_arm.set_velocity(j, v.get(j, 0.0), a_r=ACC_WRIST)
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
def map_gamepad(gp: XboxState, state: dict, dpad_to_wrist: bool) -> dict:
    """
    Args:
        gp:               XboxState (current gamepad state)
        state:            current robot joint readings
        dpad_to_wrist:    True → D-pad controls wrist pitch+roll
                          False → D-pad controls head pan+tilt
    """
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

    # ─ D-pad: wrist pitch+roll OR head pan+tilt (toggleable) ─────────
    v_wp = v_wr = 0.0
    v_hp = v_ht = 0.0
    if dpad_to_wrist:
        v_wp = gp.dpad_y * MAX_WRIST_V * p_scale          # up=+pitch
        v_wr = gp.dpad_x * MAX_WRIST_V * p_scale          # right=+roll
    else:
        v_hp = gp.dpad_x * MAX_HEAD_V * p_scale           # right=+pan
        v_ht = gp.dpad_y * MAX_HEAD_V * p_scale           # up=+tilt

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
                  "leader": "robot_gamepad", "fps": args.fps},
        fps=args.fps,
    )
    console = ConsoleInput()

    recording  = False
    episode_idx = 0
    next_t = time.time()
    frame_i = 0
    last_print = time.time()

    # Daemon-style: D-pad function toggles between wrist (default) and head
    dpad_to_wrist = True

    print("READY.  Type 's' (start recording) or use gamepad to teleop.")
    print("    button cheat-sheet:")
    print("      A=close-grip   B=open-grip   X=toggle dpad(wrist↔head)")
    print("      Y=save success   BACK=info   START=quit")
    print("      LB/RB=wrist_yaw   L2=precision   R2=fast-base")
    print()

    try:
        while True:
            t_loop = time.time()

            # 1) Read full state (encoders + odom)
            state = motor.read_state()

            # 2) Read gamepad → per-joint velocities
            vels = map_gamepad(gp, state, dpad_to_wrist)

            # 3) Send to motors
            motor.send_velocities(vels)

            # 4) Read cameras (always; cheap if no record)
            head_rgb, head_depth, ee_rgb, ee_depth = cams.read()

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
                _zh = np.zeros((CAM_RGB_H,   CAM_RGB_W,   3), dtype=np.uint8)
                _zd = np.zeros((CAM_DEPTH_H, CAM_DEPTH_W),    dtype=np.uint16)
                try:
                    # FileDataRecorder.add() signature on this robot:
                    # (ee_rgb, ee_depth, xyz, quaternion, gripper,
                    #  ee_pos, ee_rot, observations, actions,
                    #  head_rgb=None, head_depth=None)
                    recorder.add(
                        ee_rgb       = ee_rgb       if ee_rgb       is not None else _zh,
                        ee_depth     = ee_depth     if ee_depth     is not None else _zd,
                        xyz          = np.zeros(3, dtype=np.float32),     # AR-marker N/A
                        quaternion   = np.array([0,0,0,1], dtype=np.float32),
                        gripper      = float(state["gripper"]),
                        ee_pos       = np.zeros(3, dtype=np.float32),     # FK at ETL time
                        ee_rot       = np.eye(3, dtype=np.float32),
                        observations = obs_dict,
                        actions      = tgt,
                        head_rgb     = head_rgb     if head_rgb     is not None else _zh,
                        head_depth   = head_depth   if head_depth   is not None else _zd,
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
            if gp.edge_pressed("X"):           # toggle D-pad function
                dpad_to_wrist = not dpad_to_wrist
                tag = "wrist (pitch+roll)" if dpad_to_wrist else "head (pan+tilt)"
                print(f"\n[mode] D-pad → {tag}\n")
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
