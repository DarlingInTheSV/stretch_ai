#!/usr/bin/env python3
"""
robot_replay.py — replay a recorded episode on the real robot.

Reads <episode_dir>/labels.json and plays back the commanded actions
on the physical Stretch SE3.  Useful for:
  - Verifying that captured action data is sufficient to reproduce
    the original demonstration.
  - Sanity-checking the action representation.
  - Comparing different replay modes:
      --mode velocity  (re-emit recorded raw velocities; matches the
                        leader's control style during recording)
      --mode target    (use recorded absolute joint targets; smoother
                        but less faithful to original timing)

Usage on robot:
  python3 robot_replay.py /home/hello-robot/stretch_data/<.../<timestamp>/
  python3 robot_replay.py <dir> --mode target
  python3 robot_replay.py <dir> --no_prompt --fps 15
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# stretch_body needs these at IMPORT TIME under non-interactive ssh
os.environ.setdefault("HELLO_FLEET_PATH", "/home/hello-robot/stretch_user")
os.environ.setdefault("HELLO_FLEET_ID",   "stretch-se3-3139")

import stretch_body.robot as rb


# Match constants from recorder
ACC_LIFT    = 0.20
ACC_ARM     = 0.20
ACC_WRIST   = 8.0
ACC_HEAD    = 8.0
ACC_GRIPPER = 4.0
GRIPPER_MOVE_BY_PCT = 60.0


def stop_all(robot):
    """Send zero velocities to every motor + flush."""
    try:
        robot.lift.set_velocity(0.0, a_m=ACC_LIFT)
        robot.arm.set_velocity (0.0, a_m=ACC_ARM)
        robot.base.set_velocity(0.0, 0.0)
        for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            robot.end_of_arm.set_velocity(j, 0.0, a_r=ACC_WRIST)
        for j in ("head_pan", "head_tilt"):
            robot.head.set_velocity(j, 0.0, a_r=ACC_HEAD)
        robot.push_command()
    except Exception as e:
        print(f"[!] stop_all error: {e}")


def move_to_start_pose(robot, action):
    """Move arm + head to first frame's targets before replaying."""
    print("[start] moving to recorded start pose ...")
    robot.lift.move_to(action["joint_lift"])
    robot.arm.move_to (action["joint_arm_l0"])
    robot.end_of_arm.get_joint("wrist_yaw" ).move_to(action["joint_wrist_yaw"])
    robot.end_of_arm.get_joint("wrist_pitch").move_to(action["joint_wrist_pitch"])
    robot.end_of_arm.get_joint("wrist_roll" ).move_to(action["joint_wrist_roll"])
    robot.head.get_joint("head_pan" ).move_to(action["joint_head_pan"])
    robot.head.get_joint("head_tilt").move_to(action["joint_head_tilt"])
    robot.push_command()
    time.sleep(3.0)  # let trapezoidal motion settle


def _read_live_state(robot):
    """Snapshot current state from stretch_body (matching what was recorded)."""
    eoa = robot.end_of_arm.motors
    h   = robot.head.motors
    b   = robot.base.status
    return {
        "base_x":      b["x"],
        "base_y":      b["y"],
        "base_theta":  b["theta"],
        "lift":        robot.lift.status["pos"],
        "arm":         robot.arm.status["pos"],
        "wrist_yaw":   eoa["wrist_yaw"].status["pos"],
        "wrist_pitch": eoa["wrist_pitch"].status["pos"],
        "wrist_roll":  eoa["wrist_roll"].status["pos"],
        "gripper":     eoa["stretch_gripper"].status["pos"],
        "head_pan":    h["head_pan"].status["pos"],
        "head_tilt":   h["head_tilt"].status["pos"],
    }


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _diff_state(saved, live):
    """Return per-field abs error.  base_theta wraps to ±π."""
    diffs = {}
    for k in saved:
        if k not in live:
            continue
        if k == "base_theta":
            diffs[k] = abs(_wrap_pi(saved[k] - live[k]))
        else:
            diffs[k] = abs(saved[k] - live[k])
    return diffs


def replay_velocity(robot, frames, fps):
    """Replay using each frame's recorded raw velocities (v_*).

    Also samples the LIVE state each frame and compares to the recorded
    obs at the same step — useful for measuring how faithfully the
    replay reproduces the original trajectory.
    """
    dt = 1.0 / fps
    next_t = time.time()
    divergence_log = []     # list of dicts {field: abs_err}

    for i, fr in enumerate(frames):
        a = fr["actions"]
        s_saved = fr["observations"]
        # Hello-Motor lift / arm
        robot.lift.set_velocity(a.get("v_lift", 0.0), a_m=ACC_LIFT)
        robot.arm.set_velocity (a.get("v_arm",  0.0), a_m=ACC_ARM)
        # Base
        robot.base.set_velocity(a.get("v_base_forward", 0.0),
                                a.get("v_base_yaw",     0.0))
        # Wrist Dynamixels
        for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            robot.end_of_arm.set_velocity(j, a.get(f"v_{j}", 0.0), a_r=ACC_WRIST)
        # Head Dynamixels
        for j in ("head_pan", "head_tilt"):
            robot.head.set_velocity(j, a.get(f"v_{j}", 0.0), a_r=ACC_HEAD)
        # Gripper move_by (matches recording style)
        vg = a.get("v_gripper", 0.0)
        if abs(vg) > 1e-3:
            try:
                pct = math.copysign(GRIPPER_MOVE_BY_PCT, vg)
                robot.end_of_arm.get_joint("stretch_gripper").move_by(pct)
            except Exception as e:
                print(f"[!] gripper move_by failed: {e}")

        robot.push_command()

        # Sample LIVE state for divergence check
        try:
            s_live = _read_live_state(robot)
            divergence_log.append(_diff_state(s_saved, s_live))
        except Exception:
            pass

        # 1Hz progress print w/ divergence
        if i % fps == 0:
            t = i * dt
            d = divergence_log[-1] if divergence_log else {}
            print(f"  [{t:6.1f}s] frame {i:5d}/{len(frames)}  "
                  f"lift={s_saved['lift']:.2f}(Δ{d.get('lift',0):.3f}) "
                  f"arm={s_saved['arm']:.2f}(Δ{d.get('arm',0):.3f}) "
                  f"base=({s_saved['base_x']:+.2f},{s_saved['base_y']:+.2f},{s_saved['base_theta']:+.2f}) "
                  f"Δbase=({d.get('base_x',0):.3f},{d.get('base_y',0):.3f},{d.get('base_theta',0):.3f})")

        next_t += dt
        slack = next_t - time.time()
        if slack > 0:
            time.sleep(slack)
        else:
            next_t = time.time()
    return divergence_log


def replay_target(robot, frames, fps):
    """Replay using each frame's recorded absolute targets (joint_*)."""
    dt = 1.0 / fps
    next_t = time.time()
    divergence_log = []

    for i, fr in enumerate(frames):
        a = fr["actions"]
        s_saved = fr["observations"]
        # Arm chain — absolute position targets
        robot.lift.move_to(a["joint_lift"])
        robot.arm.move_to (a["joint_arm_l0"])
        robot.end_of_arm.get_joint("wrist_yaw" ).move_to(a["joint_wrist_yaw"])
        robot.end_of_arm.get_joint("wrist_pitch").move_to(a["joint_wrist_pitch"])
        robot.end_of_arm.get_joint("wrist_roll" ).move_to(a["joint_wrist_roll"])
        robot.head.get_joint("head_pan" ).move_to(a["joint_head_pan"])
        robot.head.get_joint("head_tilt").move_to(a["joint_head_tilt"])
        # Base — convert delta back to velocity for this dt
        robot.base.set_velocity(a.get("delta_s_robot", 0.0) / dt,
                                a.get("delta_theta",   0.0) / dt)
        # Gripper — same as velocity mode (move_by)
        vg = a.get("v_gripper", 0.0)
        if abs(vg) > 1e-3:
            try:
                pct = math.copysign(GRIPPER_MOVE_BY_PCT, vg)
                robot.end_of_arm.get_joint("stretch_gripper").move_by(pct)
            except Exception as e:
                print(f"[!] gripper move_by failed: {e}")

        robot.push_command()

        try:
            s_live = _read_live_state(robot)
            divergence_log.append(_diff_state(s_saved, s_live))
        except Exception:
            pass

        if i % fps == 0:
            t = i * dt
            d = divergence_log[-1] if divergence_log else {}
            print(f"  [{t:6.1f}s] frame {i:5d}/{len(frames)}  "
                  f"lift={s_saved['lift']:.2f}(Δ{d.get('lift',0):.3f}) "
                  f"arm={s_saved['arm']:.2f}(Δ{d.get('arm',0):.3f}) "
                  f"base=({s_saved['base_x']:+.2f},{s_saved['base_y']:+.2f},{s_saved['base_theta']:+.2f}) "
                  f"Δbase=({d.get('base_x',0):.3f},{d.get('base_y',0):.3f},{d.get('base_theta',0):.3f})")

        next_t += dt
        slack = next_t - time.time()
        if slack > 0:
            time.sleep(slack)
        else:
            next_t = time.time()
    return divergence_log


def _print_divergence_report(divergence_log):
    """Aggregate per-field statistics from per-frame divergences."""
    if not divergence_log:
        print("[stats] no divergence samples collected")
        return

    fields = sorted(divergence_log[0].keys())
    n = len(divergence_log)

    print("\n" + "=" * 72)
    print("STATE DIVERGENCE: |saved − live| per field")
    print("=" * 72)
    print(f"  {'field':18s}  {'mean':>10s}  {'median':>10s}  {'p95':>10s}  {'max':>10s}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    import statistics as st
    for f in fields:
        vals = [d[f] for d in divergence_log if f in d]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        med = vals_sorted[len(vals_sorted) // 2]
        p95 = vals_sorted[int(0.95 * len(vals_sorted))]
        mx = vals_sorted[-1]
        unit = "rad" if f in ("base_theta","wrist_yaw","wrist_pitch","wrist_roll",
                              "head_pan","head_tilt") else \
               "m"   if f in ("base_x","base_y","lift","arm") else ""
        print(f"  {f:18s}  {mean:10.4f}  {med:10.4f}  {p95:10.4f}  {mx:10.4f}  {unit}")
    print("=" * 72)
    print(f"  {n} samples over {n / 15:.1f}s ({len(fields)} fields)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir",
                    help="path to episode directory containing labels.json")
    ap.add_argument("--mode", choices=["velocity", "target"], default="velocity",
                    help="velocity = re-emit recorded v_* (matches recording); "
                         "target = use recorded absolute joint_* targets.")
    ap.add_argument("--fps", type=int, default=15,
                    help="replay rate (must match recording for true playback)")
    ap.add_argument("--no_prompt", action="store_true",
                    help="skip safety confirmation prompt")
    ap.add_argument("--skip_start_pose", action="store_true",
                    help="don't pre-move to recorded start pose (use current)")
    args = ap.parse_args()

    ep = Path(args.episode_dir)
    labels = ep / "labels.json"
    if not labels.exists():
        print(f"[!] {labels} not found"); sys.exit(1)

    with open(labels) as f:
        d = json.load(f)
    n = len(d)
    frames = [d[str(i)] for i in range(n)]
    duration = n / args.fps
    print(f"\nepisode: {ep}")
    print(f"  frames:    {n}")
    print(f"  duration:  {duration:.1f}s @ {args.fps} Hz")
    print(f"  mode:      {args.mode}")
    print(f"  leader:    {frames[0]['actions'].get('leader','unknown')}")

    # Show first/last action snapshot
    print(f"\nstart action: lift={frames[0]['actions']['joint_lift']:.3f} "
          f"arm={frames[0]['actions']['joint_arm_l0']:.3f} "
          f"head=({frames[0]['actions']['joint_head_pan']:+.2f},"
          f"{frames[0]['actions']['joint_head_tilt']:+.2f})")
    print(f"end   action: lift={frames[-1]['actions']['joint_lift']:.3f} "
          f"arm={frames[-1]['actions']['joint_arm_l0']:.3f} "
          f"head=({frames[-1]['actions']['joint_head_pan']:+.2f},"
          f"{frames[-1]['actions']['joint_head_tilt']:+.2f})")

    print("\n⚠️  REPLAY WILL MOVE THE ROBOT through the recorded trajectory.")
    print("    Make sure: no people in 1m, no obstacles in front,")
    print("               gripper has nothing dangerous in it,")
    print("               you can reach runstop quickly.")
    if not args.no_prompt:
        ans = input("\nProceed? (y/N): ").strip().lower()
        if ans != "y":
            print("aborted.")
            return

    # Connect
    print("\nstarting stretch_body ...")
    robot = rb.Robot()
    if not robot.startup():
        print("[!] robot.startup() returned False"); return
    if not robot.is_calibrated():
        print("[!] robot is NOT homed.  Run stretch_robot_home.py first.")
        robot.stop(); return

    try:
        if not args.skip_start_pose:
            move_to_start_pose(robot, frames[0]["actions"])

        print(f"\nreplaying ...")
        if args.mode == "velocity":
            divergence = replay_velocity(robot, frames, args.fps)
        else:
            divergence = replay_target(robot, frames, args.fps)
        print("\n[done] replay complete.")
        _print_divergence_report(divergence)
    except KeyboardInterrupt:
        print("\n[!] interrupted by user")
    finally:
        stop_all(robot)
        robot.stop()


if __name__ == "__main__":
    main()
