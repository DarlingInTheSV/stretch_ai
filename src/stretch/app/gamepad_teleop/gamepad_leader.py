# Copyright (c) Hello Robot, Inc. & lsy
# Gamepad-based teleoperation leader for Stretch SE3.
"""
Gamepad-based teleoperation leader for Stretch SE3 with data recording.

Records data compatible with stretch_ai's FileDataRecorder format
(same as dex_teleop), so downstream conversion tools work unchanged.

Usage
-----
Two gamepad source modes:

  (A) local — gamepad plugged into the 5090 leader machine via USB:
      python -m stretch.app.gamepad_teleop.gamepad_leader \
          --gamepad_source local --task pickup_cup ...

  (B) network — gamepad paired with the robot (e.g. via Bluetooth);
                a relay process runs on the robot.  This is the
                preferred mode when the leader (5090) is in a server
                room while the operator is in the same room as robot:

      # On robot:
      python3 ~/stretch_ai/src/stretch/app/gamepad_teleop/gamepad_relay.py
      # On 5090 (default --gamepad_source=network):
      python -m stretch.app.gamepad_teleop.gamepad_leader \
          --robot_ip 10.100.66.215 --task pickup_cup ...

Controls (Xbox layout)
----------------------
    Left stick  Y          base forward / backward
    Left stick  X          base rotate (yaw)
    Right stick X          arm extend / retract           (default mode)
    Right stick Y          lift up / down                 (default mode)
    LB hold + Right X      wrist roll                     (wrist mode)
    LB hold + Right Y      wrist pitch                    (wrist mode)
    RB hold + Right X      head pan                       (head mode)
    RB hold + Right Y      head tilt                      (head mode)
    D-pad  L/R             wrist yaw  (always)
    L2 trigger             gripper close
    R2 trigger             gripper open

    A button               toggle recording on / off (no save on stop)
    Y button               save current episode as SUCCESS
    B button               abandon current episode
    START button           quit
    BACK button            print current state (debug)
"""

import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.utils.data_tools.record import FileDataRecorder

try:
    import evdev
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False

import json
import zmq


# ── Robot joint limits (Stretch SE3) ──────────────────────────────────
LIFT_MIN,  LIFT_MAX  = 0.15,  1.10
ARM_MIN,   ARM_MAX   = 0.00,  0.50
WRIST_YAW_MIN,   WRIST_YAW_MAX   = -1.50, 1.50
WRIST_PITCH_MIN, WRIST_PITCH_MAX = -1.40, 0.40
WRIST_ROLL_MIN,  WRIST_ROLL_MAX  = -2.00, 2.00
HEAD_PAN_MIN,    HEAD_PAN_MAX    = -2.50, 2.50
HEAD_TILT_MIN,   HEAD_TILT_MAX   = -1.50, 0.50

GRIPPER_CLOSED = -0.30
GRIPPER_OPEN   =  0.60

# ── Velocity scaling at full stick deflection ─────────────────────────
MAX_BASE_V    = 0.30    # m/s   forward
MAX_BASE_W    = 0.80    # rad/s yaw
MAX_LIFT_V    = 0.20    # m/s   (was 0.10, felt sluggish)
MAX_ARM_V     = 0.15    # m/s   (was 0.10)
MAX_WRIST_V   = 1.50    # rad/s
MAX_HEAD_V    = 1.50    # rad/s
MAX_GRIPPER_V = 3.00    # gripper-units/s ([-0.3,0.6] range, was 1.5 too slow)

STICK_DEADZONE   = 0.10
TRIGGER_DEADZONE = 0.05


# ── Small math helpers ────────────────────────────────────────────────
def deadzone(v, dz=STICK_DEADZONE):
    if abs(v) < dz:
        return 0.0
    return math.copysign((abs(v) - dz) / (1.0 - dz), v)


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ══ Gamepad reader (async, evdev) ════════════════════════════════════
class XboxController:
    """Async Xbox-style gamepad reader. Works with USB or BT pads on Linux."""

    def __init__(self, device_path: Optional[str] = None):
        if not HAS_EVDEV:
            raise ImportError("evdev not installed.  pip install evdev")

        if device_path is None:
            self.dev = self._auto_detect()
        else:
            self.dev = evdev.InputDevice(device_path)

        print(f"[Gamepad] using {self.dev.name!r} at {self.dev.path}")

        # Cache axis ranges for normalization
        self._abs_info = {}
        for code, info in self.dev.capabilities().get(evdev.ecodes.EV_ABS, []):
            self._abs_info[code] = info

        # Axis state ([-1,+1] for sticks, [0,1] for triggers)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        self.lt = 0.0
        self.rt = 0.0
        self.dpad_x = 0    # -1, 0, 1
        self.dpad_y = 0

        # Button state (held = 1)
        self.btn = {k: 0 for k in
                    ("A", "B", "X", "Y", "LB", "RB", "START", "BACK", "LSTICK", "RSTICK")}
        self._prev_btn = dict(self.btn)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _auto_detect():
        devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
        keywords = ("xbox", "x-box", "gamepad", "controller", "joystick",
                    "playstation", "ds4", "dualshock", "dualsense",
                    "8bitdo", "stadia", " pad")
        excludes = ("touchscreen", "touchpad", "passthrough", "trackpad")
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
            available = "\n  ".join(f"{d.path}: {d.name}" for d in devs) or "(none)"
            raise RuntimeError(
                "No gamepad detected.  Available evdev devices:\n  " + available)
        return prio[0]

    def _norm_axis(self, code, raw):
        info = self._abs_info.get(code)
        if not info:
            return 0.0
        # Triggers: 0..max → 0..1
        if code in (evdev.ecodes.ABS_Z, evdev.ecodes.ABS_RZ):
            return raw / info.max if info.max > 0 else 0.0
        # Sticks: min..max → -1..+1
        mid = 0.5 * (info.min + info.max)
        half = 0.5 * (info.max - info.min)
        return (raw - mid) / half if half > 0 else 0.0

    def _loop(self):
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
                v = self._norm_axis(ev.code, ev.value)
                if   ev.code == evdev.ecodes.ABS_X:   self.left_x = v
                elif ev.code == evdev.ecodes.ABS_Y:   self.left_y = -v        # flip: stick up = +1
                elif ev.code == evdev.ecodes.ABS_RX:  self.right_x = v
                elif ev.code == evdev.ecodes.ABS_RY:  self.right_y = -v
                elif ev.code == evdev.ecodes.ABS_Z:   self.lt = v
                elif ev.code == evdev.ecodes.ABS_RZ:  self.rt = v
                elif ev.code == evdev.ecodes.ABS_HAT0X: self.dpad_x = int(np.sign(ev.value))
                elif ev.code == evdev.ecodes.ABS_HAT0Y: self.dpad_y = int(np.sign(-ev.value))

            elif ev.type == evdev.ecodes.EV_KEY:
                name = BTN_MAP.get(ev.code)
                if name is not None:
                    self.btn[name] = ev.value

    def edge_pressed(self, name):
        """Return True only on rising edge (button press)."""
        was = self._prev_btn.get(name, 0)
        now = self.btn.get(name, 0)
        edge = (was == 0 and now == 1)
        self._prev_btn[name] = now
        return edge

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


# ══ Network gamepad reader (subscribes to gamepad_relay on robot) ════
class NetworkXboxController:
    """Subscribes to a remote gamepad_relay over ZMQ.

    Same interface as XboxController so it's drop-in compatible.
    """

    def __init__(self, sub_url: str = "tcp://10.100.66.215:4499",
                 stale_warn_s: float = 2.0):
        self.sub_url = sub_url
        self.stale_warn_s = stale_warn_s

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)        # only keep latest msg
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(sub_url)
        print(f"[Gamepad/net] SUB connected to {sub_url}")

        # State (same fields as XboxController)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        self.lt = 0.0
        self.rt = 0.0
        self.dpad_x = 0
        self.dpad_y = 0
        self.btn = {k: 0 for k in
                    ("A", "B", "X", "Y", "LB", "RB", "START", "BACK",
                     "LSTICK", "RSTICK")}
        self._prev_btn = dict(self.btn)
        self.last_msg_ts = 0.0
        self.last_warn_ts = 0.0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=200))
            if self.sock in socks:
                try:
                    raw = self.sock.recv_string(zmq.NOBLOCK)
                    snap = json.loads(raw)
                except Exception as e:
                    print(f"[Gamepad/net] recv error: {e}")
                    continue
                self.left_x  = float(snap.get("left_x",  0.0))
                self.left_y  = float(snap.get("left_y",  0.0))
                self.right_x = float(snap.get("right_x", 0.0))
                self.right_y = float(snap.get("right_y", 0.0))
                self.lt      = float(snap.get("lt",      0.0))
                self.rt      = float(snap.get("rt",      0.0))
                self.dpad_x  = int(snap.get("dpad_x",  0))
                self.dpad_y  = int(snap.get("dpad_y",  0))
                self.btn.update(snap.get("btn", {}))
                self.last_msg_ts = time.time()

            # Periodic stale-link warning
            now = time.time()
            if (self.last_msg_ts > 0
                and now - self.last_msg_ts > self.stale_warn_s
                and now - self.last_warn_ts > 5.0):
                print(f"[Gamepad/net] WARNING no relay msgs for "
                      f"{now - self.last_msg_ts:.1f}s")
                self.last_warn_ts = now

    def edge_pressed(self, name):
        was = self._prev_btn.get(name, 0)
        now = self.btn.get(name, 0)
        edge = (was == 0 and now == 1)
        self._prev_btn[name] = now
        return edge

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close()


# ══ Main leader class ════════════════════════════════════════════════
class GamepadLeader:
    def __init__(
        self,
        robot_ip: str,
        task: str,
        user: str,
        env: str,
        data_dir: str,
        fps: int = 15,
        gamepad_device: Optional[str] = None,
        gamepad_source: str = "local",   # "local" or "network"
        gamepad_url: Optional[str] = None,  # used if gamepad_source=="network"
        save_images: bool = False,
    ):
        self.fps = fps
        self.dt = 1.0 / fps

        # ── Robot ─────────────────────────────────────────────────────
        print(f"[Leader] connecting to {robot_ip} ...")
        self.robot = HomeRobotZmqClient(
            robot_ip=robot_ip, enable_rerun_server=False
        )
        self.robot.start()
        time.sleep(3.0)
        if not self.robot.is_homed:
            raise SystemExit(
                "[ERROR] Robot is not homed.  Run stretch_robot_home.py first.")
        print(f"[Leader] robot OK, in_manip={self.robot.in_manipulation_mode()}, "
              f"in_nav={self.robot.in_navigation_mode()}")
        # We stay in manipulation mode for arm + gripper + head;
        # base velocity will go through set_velocity which works in either mode.
        self.robot.switch_to_manipulation_mode()
        time.sleep(1.0)

        # ── Gamepad ───────────────────────────────────────────────────
        if gamepad_source == "network":
            url = gamepad_url or f"tcp://{robot_ip}:4499"
            self.gp = NetworkXboxController(url)
        else:
            self.gp = XboxController(gamepad_device)

        # ── Recorder ──────────────────────────────────────────────────
        self.task, self.user, self.env = task, user, env
        self.recorder = FileDataRecorder(
            datadir=data_dir,
            task=task, user=user, env=env,
            save_images=save_images,
            metadata={
                "task": task, "user": user, "env": env,
                "leader": "gamepad",
                "fps": fps,
            },
            fps=fps,
        )

        # ── Velocity EMA for base ─────────────────────────────────────
        self.prev_base_pose = self.robot.get_base_pose().copy()
        self.prev_t = time.time()
        self.v_fwd_ema = 0.0
        self.v_yaw_ema = 0.0

        # ── Recording state ───────────────────────────────────────────
        self.recording = False
        self.episode_idx = 0

        print("\n[Leader] ready.")
        print("  A     = toggle record (no save on stop)")
        print("  Y     = save SUCCESS")
        print("  B     = abandon episode")
        print("  START = quit")
        print("  BACK  = print current state\n")

    # ── State / target computation ────────────────────────────────────
    def read_state(self):
        joints = self.robot.get_joint_positions()
        base_pose = self.robot.get_base_pose().copy()  # (x, y, theta)
        # Base velocity: finite-diff + EMA (until bridge exposes /odom twist)
        now_t = time.time()
        dt = max(now_t - self.prev_t, 1e-3)
        dx = base_pose[0] - self.prev_base_pose[0]
        dy = base_pose[1] - self.prev_base_pose[1]
        cos_t = math.cos(self.prev_base_pose[2])
        sin_t = math.sin(self.prev_base_pose[2])
        v_fwd_raw = (dx * cos_t + dy * sin_t) / dt
        v_yaw_raw = wrap_to_pi(base_pose[2] - self.prev_base_pose[2]) / dt
        alpha = 0.2
        self.v_fwd_ema = (1 - alpha) * self.v_fwd_ema + alpha * v_fwd_raw
        self.v_yaw_ema = (1 - alpha) * self.v_yaw_ema + alpha * v_yaw_raw
        self.prev_base_pose = base_pose
        self.prev_t = now_t

        return {
            "base_pose": base_pose,
            "v_forward": self.v_fwd_ema,
            "v_yaw": self.v_yaw_ema,
            "lift":         joints[3],
            "arm":          joints[4],
            "gripper":      joints[5],   # raw, in [-0.3, 0.6]
            "wrist_roll":   joints[6],
            "wrist_pitch":  joints[7],
            "wrist_yaw":    joints[8],
            "head_pan":     joints[9],
            "head_tilt":    joints[10],
        }

    def compute_targets(self, state):
        gp = self.gp
        wrist_mode = bool(gp.btn["LB"])
        head_mode  = bool(gp.btn["RB"])

        # Base from left stick
        v_fwd = deadzone(gp.left_y) * MAX_BASE_V
        v_yaw = -deadzone(gp.left_x) * MAX_BASE_W   # stick LEFT → +yaw (CCW)

        rx = deadzone(gp.right_x)
        ry = deadzone(gp.right_y)

        # Right stick: 3 modes
        v_lift = v_arm = v_wrist_pitch = v_wrist_roll = 0.0
        v_head_pan = v_head_tilt = 0.0
        if head_mode:
            v_head_pan  = -rx * MAX_HEAD_V
            v_head_tilt =  ry * MAX_HEAD_V
        elif wrist_mode:
            v_wrist_roll  = rx * MAX_WRIST_V
            v_wrist_pitch = ry * MAX_WRIST_V
        else:
            v_arm  = rx * MAX_ARM_V
            v_lift = ry * MAX_LIFT_V

        # D-pad: wrist yaw always available
        v_wrist_yaw = -gp.dpad_x * MAX_WRIST_V * 0.6

        # Triggers: gripper
        v_gripper = max(0.0, gp.rt - TRIGGER_DEADZONE) * MAX_GRIPPER_V \
                  - max(0.0, gp.lt - TRIGGER_DEADZONE) * MAX_GRIPPER_V

        # Integrate to absolute, clip to limits
        t_lift  = float(np.clip(state["lift"]  + v_lift  * self.dt, LIFT_MIN, LIFT_MAX))
        t_arm   = float(np.clip(state["arm"]   + v_arm   * self.dt, ARM_MIN, ARM_MAX))
        t_yaw   = float(np.clip(state["wrist_yaw"]   + v_wrist_yaw   * self.dt, WRIST_YAW_MIN, WRIST_YAW_MAX))
        t_pitch = float(np.clip(state["wrist_pitch"] + v_wrist_pitch * self.dt, WRIST_PITCH_MIN, WRIST_PITCH_MAX))
        t_roll  = float(np.clip(state["wrist_roll"]  + v_wrist_roll  * self.dt, WRIST_ROLL_MIN, WRIST_ROLL_MAX))
        t_grip  = float(np.clip(state["gripper"]     + v_gripper     * self.dt, GRIPPER_CLOSED, GRIPPER_OPEN))
        t_hp    = float(np.clip(state["head_pan"]    + v_head_pan    * self.dt, HEAD_PAN_MIN, HEAD_PAN_MAX))
        t_ht    = float(np.clip(state["head_tilt"]   + v_head_tilt   * self.dt, HEAD_TILT_MIN, HEAD_TILT_MAX))

        return {
            "joint_lift":         t_lift,
            "joint_arm_l0":       t_arm,
            "joint_wrist_yaw":    t_yaw,
            "joint_wrist_pitch":  t_pitch,
            "joint_wrist_roll":   t_roll,
            "stretch_gripper":    t_grip,
            "joint_head_pan":     t_hp,
            "joint_head_tilt":    t_ht,
            # Base — sent as velocity, but stored for record
            "base_v_forward":     v_fwd,
            "base_v_yaw":         v_yaw,
        }

    # ── Sending to robot ──────────────────────────────────────────────
    def send(self, targets):
        # Arm + wrist + gripper + head via arm_to (manipulation mode)
        # arm_to internally builds {"joint": [...], "gripper": ..., "head_to": ...}
        # which matches the bridge's `elif "joint" in action` branch.
        joint_angles = np.array([
            0.0,                                   # base_x_joint unused
            targets["joint_lift"],
            targets["joint_arm_l0"],
            targets["joint_wrist_yaw"],
            targets["joint_wrist_pitch"],
            targets["joint_wrist_roll"],
        ])
        head = np.array([targets["joint_head_pan"],
                         targets["joint_head_tilt"]])
        try:
            self.robot.arm_to(
                joint_angles=joint_angles,
                gripper=targets["stretch_gripper"],
                head=head,
                blocking=False,
                reliable=False,
            )
        except Exception as e:
            print(f"[Leader] arm_to failed: {e}")

        # Base velocity — must use set_base_velocity (NOT set_velocity).
        # set_base_velocity sends {"base_velocity": {"v":..., "w":...}}, the
        # only key the bridge action handler actually matches.
        try:
            self.robot.set_base_velocity(
                targets["base_v_forward"], targets["base_v_yaw"]
            )
        except Exception as e:
            print(f"[Leader] set_base_velocity failed: {e}")

    # ── Recording ─────────────────────────────────────────────────────
    def record_frame(self, state, targets):
        """Add one frame to the recorder, in dex-compatible format."""
        if not self.recording:
            return
        try:
            obs = self.robot.get_observation()
        except Exception as e:
            print(f"[Leader] get_observation failed: {e}")
            return

        # head camera (RGB + depth) — mandatory
        head_rgb = getattr(obs, "rgb", None)
        head_depth = getattr(obs, "depth", None)
        head_cam_pose = getattr(obs, "camera_pose", np.eye(4))
        ee_pose = getattr(obs, "ee_pose", np.eye(4))

        # End-effector camera (gripper d405) — try servo channel
        ee_rgb, ee_depth, ee_cam_pose = None, None, np.eye(4)
        try:
            servo = self.robot.get_servo_observation()
            ee_rgb = getattr(servo, "ee_rgb", None) or getattr(servo, "rgb", None)
            ee_depth = getattr(servo, "ee_depth", None) or getattr(servo, "depth", None)
            ee_cam_pose = getattr(servo, "ee_camera_pose", np.eye(4))
        except Exception:
            pass
        # Fallback: use head camera as ee placeholder so recording doesn't break
        if ee_rgb is None:
            ee_rgb = head_rgb
            ee_depth = head_depth

        # Joint observations (HelloStretchIdx names → values)
        joint_obs = {
            "base_x":       state["base_pose"][0],
            "base_y":       state["base_pose"][1],
            "base_theta":   state["base_pose"][2],
            "lift":         state["lift"],
            "arm":          state["arm"],
            "gripper":      state["gripper"],
            "wrist_roll":   state["wrist_roll"],
            "wrist_pitch":  state["wrist_pitch"],
            "wrist_yaw":    state["wrist_yaw"],
            "head_pan":     state["head_pan"],
            "head_tilt":    state["head_tilt"],
            # Velocity (extras, not in HelloStretchIdx)
            "base_v_forward": state["v_forward"],
            "base_v_yaw":     state["v_yaw"],
        }

        # Action dict — same key names as dex teleop's goal_configuration,
        # plus head + base velocity extras for our gamepad case
        action_dict = {
            "joint_lift":         targets["joint_lift"],
            "joint_arm_l0":       targets["joint_arm_l0"],
            "joint_wrist_yaw":    targets["joint_wrist_yaw"],
            "joint_wrist_pitch":  targets["joint_wrist_pitch"],
            "joint_wrist_roll":   targets["joint_wrist_roll"],
            "stretch_gripper":    targets["stretch_gripper"],
            "joint_head_pan":     targets["joint_head_pan"],
            "joint_head_tilt":    targets["joint_head_tilt"],
            # Base — both representations kept, ETL picks one
            "base_v_forward":     targets["base_v_forward"],
            "base_v_yaw":         targets["base_v_yaw"],
            # Mark as gamepad data
            "leader":             "gamepad",
        }

        # FileDataRecorder.add() — fill AR-marker fields with zeros (gamepad has no markers)
        zero3 = np.zeros(3, dtype=np.float32)
        zero4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identity quat

        try:
            self.recorder.add(
                ee_rgb=ee_rgb,
                ee_depth=ee_depth,
                ee_cam_pose=np.asarray(ee_cam_pose),
                xyz=zero3,                 # AR-marker target xyz — N/A for gamepad
                quaternion=zero4,          # AR-marker target quat — N/A for gamepad
                gripper=float(targets["stretch_gripper"]),
                ee_pose=np.asarray(ee_pose),
                observations=joint_obs,
                actions=action_dict,
                head_rgb=head_rgb,
                head_depth=head_depth,
                head_cam_pose=np.asarray(head_cam_pose),
            )
        except Exception as e:
            print(f"[Leader] recorder.add failed: {e}")

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self):
        next_t = time.time()
        try:
            while True:
                if self.gp.edge_pressed("START"):
                    print("[Leader] STOP requested.")
                    break

                # 1. read state
                state = self.read_state()

                # 2. read gamepad → compute targets
                targets = self.compute_targets(state)

                # 3. send to robot
                self.send(targets)

                # 4. handle button events
                if self.gp.edge_pressed("A"):
                    if not self.recording:
                        self.episode_idx += 1
                        print(f"\n[Leader] >>> recording START  (episode #{self.episode_idx})")
                        self.recording = True
                    else:
                        print("[Leader] <<< recording PAUSED (press Y to save / B to abandon)")
                        self.recording = False

                if self.gp.edge_pressed("Y"):
                    if self.recorder.step > 0:
                        n = self.recorder.step
                        print(f"[Leader] ✓ saving SUCCESS episode  ({n} frames)")
                        self.recording = False
                        self.recorder.write(success=True)
                    else:
                        print("[Leader] (Y) no frames to save")

                if self.gp.edge_pressed("B"):
                    if self.recorder.step > 0:
                        print(f"[Leader] ✗ abandoning episode  ({self.recorder.step} frames discarded)")
                    self.recorder.reset()
                    self.recording = False

                if self.gp.edge_pressed("BACK"):
                    self._print_state(state, targets)

                # 5. record frame if active
                if self.recording:
                    self.record_frame(state, targets)

                # 6. pacing
                next_t += self.dt
                slack = next_t - time.time()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_t = time.time()
        finally:
            print("[Leader] shutdown ...")
            try:
                self.robot.set_velocity(0.0, 0.0)
            except Exception:
                pass
            self.gp.stop()
            self.robot.stop()

    def _print_state(self, state, targets):
        print("\n── STATE ──")
        print(f"  base_pose = ({state['base_pose'][0]:+.3f}, "
              f"{state['base_pose'][1]:+.3f}, {state['base_pose'][2]:+.3f})")
        print(f"  base_vel  = (fwd={state['v_forward']:+.3f}, yaw={state['v_yaw']:+.3f})")
        print(f"  lift={state['lift']:.3f}  arm={state['arm']:.3f}  "
              f"yaw={state['wrist_yaw']:+.2f} pitch={state['wrist_pitch']:+.2f} "
              f"roll={state['wrist_roll']:+.2f}  grip={state['gripper']:+.2f}")
        print(f"  head=(pan={state['head_pan']:+.2f}, tilt={state['head_tilt']:+.2f})")
        print("── TARGETS ──")
        for k, v in targets.items():
            print(f"  {k:20s} = {v:+.3f}")
        print("── REC ──")
        print(f"  recording={self.recording}, frames={self.recorder.step}\n")


# ══ CLI ══════════════════════════════════════════════════════════════
@click.command()
@click.option("--robot_ip",       default="10.100.66.215", help="Stretch robot IP")
@click.option("--task",           default="default_task",   help="task name (folder)")
@click.option("--user",           default="default_user",   help="user name (folder)")
@click.option("--env",            default="default_env",    help="env name  (folder)")
@click.option("--data_dir",       default="./data",         help="root dir for recordings")
@click.option("--fps",            default=15,               help="control + record rate (Hz)")
@click.option("--gamepad_source", type=click.Choice(["local", "network"]),
              default="network",
              help="gamepad input source: 'local' (USB to 5090) or "
                   "'network' (relay running on robot)")
@click.option("--gamepad_url",    default=None,
              help="ZMQ SUB URL for network gamepad relay "
                   "(default: tcp://<robot_ip>:4499)")
@click.option("--gamepad_device", default=None,
              help="evdev path (only for --gamepad_source local)")
@click.option("--save_images",    is_flag=True,             help="keep raw png frames (debug)")
def main(robot_ip, task, user, env, data_dir, fps,
         gamepad_source, gamepad_url, gamepad_device, save_images):
    leader = GamepadLeader(
        robot_ip=robot_ip,
        task=task,
        user=user,
        env=env,
        data_dir=data_dir,
        fps=fps,
        gamepad_source=gamepad_source,
        gamepad_url=gamepad_url,
        gamepad_device=gamepad_device,
        save_images=save_images,
    )
    leader.run()


if __name__ == "__main__":
    main()
