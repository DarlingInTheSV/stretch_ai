#!/usr/bin/env python3
"""
Gamepad relay — runs on the Stretch robot.

Reads a paired Bluetooth/USB gamepad via evdev, publishes its state as JSON
over ZMQ PUB on tcp://0.0.0.0:4499, so a remote leader (e.g. on a 5090 in a
server room) can subscribe and use the gamepad as if it were locally
connected.

Usage on the robot:

    # If gamepad not paired yet:
    bluetoothctl
    > scan on
    > pair  <MAC>
    > trust <MAC>
    > connect <MAC>
    > exit

    # Then run the relay:
    python3 gamepad_relay.py
    # (defaults: auto-detect gamepad, bind tcp://0.0.0.0:4499, 30 Hz)
"""
import argparse
import json
import threading
import time
from typing import Optional

import evdev
import zmq


class XboxState:
    """Maintain Xbox-style gamepad state from evdev events."""

    def __init__(self, device_path: Optional[str] = None):
        if device_path is None:
            self.dev = self._auto_detect()
        else:
            self.dev = evdev.InputDevice(device_path)
        print(f"[relay] using {self.dev.name!r} at {self.dev.path}")

        self._abs_info = {}
        for code, info in self.dev.capabilities().get(evdev.ecodes.EV_ABS, []):
            self._abs_info[code] = info

        # State
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
        self.last_event_ts = 0.0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _auto_detect():
        devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
        keywords = ("xbox", "gamepad", "controller", "joystick",
                    "playstation", "ds4", "dualshock", "dualsense", "8bitdo")
        prio = [d for d in devs if any(k in d.name.lower() for k in keywords)]
        if not prio:
            prio = [
                d for d in devs
                if evdev.ecodes.EV_ABS in d.capabilities()
                and any(c == evdev.ecodes.ABS_X for c, _ in
                        d.capabilities()[evdev.ecodes.EV_ABS])
            ]
        if not prio:
            avail = "\n  ".join(f"{d.path}: {d.name}" for d in devs) or "(none)"
            raise RuntimeError(
                "No gamepad detected. Available evdev devices:\n  " + avail)
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
        BTN = {
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
            self.last_event_ts = time.time()
            if ev.type == evdev.ecodes.EV_ABS:
                v = self._norm(ev.code, ev.value)
                if   ev.code == evdev.ecodes.ABS_X:   self.left_x = v
                elif ev.code == evdev.ecodes.ABS_Y:   self.left_y = -v
                elif ev.code == evdev.ecodes.ABS_RX:  self.right_x = v
                elif ev.code == evdev.ecodes.ABS_RY:  self.right_y = -v
                elif ev.code == evdev.ecodes.ABS_Z:   self.lt = v
                elif ev.code == evdev.ecodes.ABS_RZ:  self.rt = v
                elif ev.code == evdev.ecodes.ABS_HAT0X: self.dpad_x = 1 if ev.value > 0 else (-1 if ev.value < 0 else 0)
                elif ev.code == evdev.ecodes.ABS_HAT0Y: self.dpad_y = -1 if ev.value > 0 else (1 if ev.value < 0 else 0)
            elif ev.type == evdev.ecodes.EV_KEY:
                name = BTN.get(ev.code)
                if name is not None:
                    self.btn[name] = ev.value

    def snapshot(self):
        """Return a JSON-serializable snapshot of current state."""
        return {
            "left_x":  self.left_x,
            "left_y":  self.left_y,
            "right_x": self.right_x,
            "right_y": self.right_y,
            "lt":      self.lt,
            "rt":      self.rt,
            "dpad_x":  self.dpad_x,
            "dpad_y":  self.dpad_y,
            "btn":     dict(self.btn),
            "ts":      time.time(),
            "last_event_ts": self.last_event_ts,
            "device":  self.dev.name,
        }

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser(description="Gamepad relay (robot side)")
    ap.add_argument("--bind", default="tcp://0.0.0.0:4499",
                    help="ZMQ PUB bind address")
    ap.add_argument("--rate", type=int, default=30,
                    help="publish rate (Hz)")
    ap.add_argument("--device", default=None,
                    help="evdev path; if omitted, auto-detects")
    args = ap.parse_args()

    state = XboxState(args.device)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.bind)
    print(f"[relay] PUB ready on {args.bind}, publishing at {args.rate} Hz")
    print("[relay] (ctrl-c to stop)")

    period = 1.0 / args.rate
    next_t = time.time()
    msg_count = 0
    last_log = time.time()

    try:
        while True:
            snap = state.snapshot()
            pub.send_string(json.dumps(snap))
            msg_count += 1

            # Periodic status log
            if time.time() - last_log >= 5.0:
                age = time.time() - state.last_event_ts if state.last_event_ts else float("inf")
                print(f"[relay] {msg_count} msgs sent  "
                      f"(last gamepad event {age:.1f}s ago)")
                last_log = time.time()

            next_t += period
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.time()
    except KeyboardInterrupt:
        print("\n[relay] stopping ...")
    finally:
        state.stop()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
