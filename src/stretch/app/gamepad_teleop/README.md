# Stretch SE3 Gamepad Teleop + Data Recording

End-to-end pipeline to teleoperate the Stretch SE3 with a USB/Bluetooth
gamepad and record demonstrations for VLA training (compatible with
stretch_ai's dex teleop data format and downstream tooling).

## Why this exists

The default `stretch_ros2_bridge` Docker → ZMQ → 5090 path is great
for AI inference but **adds 50–150 ms latency + position-target control**
which feels sluggish for human teleop.

This module replaces that loop during data collection only:
the gamepad → robot control → camera capture → disk all run on the
robot itself, calling `stretch_body` directly with **per-joint velocity
commands** — same architecture as Hello Robot's silky-smooth official
`stretch_gamepad_teleop` daemon.

After data is collected, switch back to the bridge for AI inference.

---

## Components

| Script | Where it runs | Purpose |
|---|---|---|
| `gamepad_relay.py` | robot | Reads gamepad via evdev, ZMQ-PUB state to network. **Used only when leader runs on 5090** (legacy). |
| `gamepad_leader.py` | 5090 | Subscribes to robot's relay or local gamepad, sends actions via stretch_ai bridge. **Slower path, kept for compatibility.** |
| `robot_leader.py` ⭐ | robot | **Primary leader.** Direct stretch_body control for silky-smooth teleop + dual RealSense capture + FileDataRecorder. |
| `robot_replay.py` | robot | Reads a saved `labels.json` and replays the actions on the physical robot. |
| `/home/hello-robot/start_bridge.sh` | robot | Starts the patched bridge container for AI-inference mode (auto-kills daemon, starts clash, starts container with --restart=unless-stopped). |

Default workflow uses `robot_leader.py` for collection + `robot_replay.py`
for playback verification.

---

## Mode switching: collection vs inference

The robot's `/dev/hello-*` hardware ports are **single-owner**.
Only one of these can run at a time:

| Mode | What's running | Hardware owner |
|---|---|---|
| **Collection** (this guide) | `robot_leader.py` directly | stretch_body in robot_leader |
| **Inference** | `start_bridge.sh` → docker container | stretch_body inside the container |
| **Owner default** | `stretch_gamepad_teleop` daemon (auto-starts) | the daemon |

`start_bridge.sh` automatically `kill`s the gamepad_teleop daemon before
launching the container.  When you stop the bridge / leave the robot,
the systemd unit will restart the daemon at next boot.

**To switch from collection to inference:**
```bash
# stop the leader (Ctrl+C in its terminal)
# then on the robot:
docker stop stretch_bridge       # if a stale container is around
/home/hello-robot/start_bridge.sh
```

**To switch from inference to collection:**
```bash
docker stop stretch_bridge       # release hardware
python3 /home/hello-robot/robot_leader.py ...
```

---

## Collection: robot_leader.py

### Quick start

On the robot (operator-side, standing next to it):

```bash
ssh hello-robot@10.100.66.215

# 1. make sure bridge container is stopped (frees hardware)
docker stop stretch_bridge

# 2. plug in / power up Xbox-style gamepad

# 3. run leader
python3 /home/hello-robot/robot_leader.py \
    --task pickup_cup \
    --user lsy \
    --env office \
    --data_dir /home/hello-robot/stretch_data \
    --fps 15 \
    --image_size 224 \
    --video_crf 15
```

Recommended flags:
- `--image_size 224` matches pi0.5 / OpenVLA SigLIP backbone input directly (vs. native 640×480 = 6× larger files).
- `--video_crf 15` near-visually-lossless h264 (vs. FileDataRecorder's
  default 30 which is blurry).
- `--save_depth` only if you specifically need depth (slow on long
  episodes, default off).

### Gamepad mapping (modeled after stretch_gamepad_teleop daemon)

| Control | Action |
|---|---|
| Left stick X | base yaw |
| Left stick Y | base forward / backward |
| Right stick X | arm extend / retract |
| Right stick Y | lift up / down |
| LB / RB | wrist_yaw + / − (always) |
| D-pad up/down | wrist_pitch + / − |
| D-pad left/right | wrist_roll + / − |
| A button (hold) | gripper close (move_by per frame, daemon-style) |
| B button (hold) | gripper open |
| L2 trigger | precision mode (×0.3 vel) while held |
| R2 trigger | fast-base mode (base × 1.5) while held |
| **X button** | toggle head pose: `'ahead'` (pan=0, tilt=−65°) ↔ `'tool'` (pan=−90°, tilt=−45°) |
| Y button | save current episode as SUCCESS |
| BACK button | print full state snapshot |
| START button | quit cleanly |
| LB+RB combo | toggle recording start/pause |

### Console commands (typed in the terminal during run)

| Command | Effect |
|---|---|
| `s` / `start` | start a new recording episode |
| `p` / `pause` | pause recording (buffer kept) |
| `y` / `save` | save current episode as SUCCESS |
| `a` / `abandon` | discard current buffer |
| `i` / `info` | print full state snapshot |
| `q` / `quit` | exit cleanly |

### Live state line

Every second the leader prints something like:
```
[REC #2/158f] base=(+0.34,+0.27,+2.10) v=(+0.01,+0.00) lift=0.72 arm=0.00 yaw=+0.07 grip=3.23 head=(-0.01,-1.10)
```
- `[REC #N/Mf]` — recording episode N, M frames buffered
- `[----]` when not recording

### Head poses (matches stretch_ai/ai_pickup conventions)

```python
'ahead': (pan=0,    tilt=-65°)   # navigation - sees floor + obstacles
'tool':  (pan=-90°, tilt=-45°)   # manipulation - centers gripper in view
```

---

## Replay: robot_replay.py

After recording, verify the data by replaying on the real robot:

```bash
docker stop stretch_bridge   # if running

python3 /home/hello-robot/robot_replay.py \
    /home/hello-robot/stretch_data/pickup_cup/lsy/office/2026-04-30--04-51-42

# or with explicit mode + skip-prompt
python3 /home/hello-robot/robot_replay.py <dir> --mode target --no_prompt
```

### Two replay modes

| Flag | Approach | Use case |
|---|---|---|
| `--mode velocity` (default) | Re-emit `v_lift`/`v_arm`/etc per frame via `set_velocity` | Most faithful to original recording control style |
| `--mode target` | `move_to()` on each absolute `joint_*` target per frame | Cleaner trajectory; tests "absolute-target" VLA inference style |

Both modes:
1. Pre-move to recorded start pose (3s settle) then begin
2. Print 1 Hz progress
3. On Ctrl+C / exception → zero all velocities + robot.stop() (safe)

### What to look for during replay

- ✅ Robot reaches roughly the same end pose as the original demo
- ✅ Gripper opens/closes at the right moments
- ✅ Base moves through similar trajectory (small drift expected from wheel-encoder odometry)
- ⚠️ If big divergence → check fps matches recording, check start pose matches

---

## Data on disk

Each saved episode produces:

```
<data_dir>/<task>/<user>/<env>/<YYYY-MM-DD--HH-MM-SS>/
├── head_compressed_video_h264.mp4         # head D435if RGB (h264 crf=15)
├── gripper_compressed_video_h264.mp4      # gripper D405 RGB
├── stitched_head_ee.mp4                   # head|gripper hstack (auto-generated)
├── compressed_np_head_depth_float32.bin   # tiny placeholder if --save_depth off
├── compressed_np_gripper_depth_float32.bin
├── labels.json                            # ⭐ per-frame state + action
├── configs.json                           # episode metadata + git info
├── completed.txt                          # marker
└── success.txt                            # only on Y / "save" command
```

### labels.json schema (per frame)

```json
{
  "0": {
    "step": 0,
    "observations": {                      // ← STATE (raw, episode-normalize at ETL)
      "base_x": -0.348, "base_y": 0.274, "base_theta": 2.099,    // odom world frame
      "lift": 0.724, "arm": 0.003,
      "wrist_yaw": 0.07, "wrist_pitch": -0.73, "wrist_roll": 0.04,
      "gripper": 3.23,                      // raw servo units, range ~[-0.45, 70+]
      "head_pan": -0.01, "head_tilt": -1.10,
      "base_v_forward": 0.0, "base_v_yaw": 0.002
    },
    "actions": {                           // ← ACTION (commanded, derived from velocity)
      // 8 absolute targets (= state + v×dt, clipped)
      "joint_lift": 0.724,
      "joint_arm_l0": 0.003,
      "joint_wrist_yaw":  0.07,
      "joint_wrist_pitch": -0.73,
      "joint_wrist_roll": 0.04,
      "stretch_gripper": 3.23,             // raw, normalize at ETL
      "joint_head_pan": -0.01,
      "joint_head_tilt": -1.10,
      // 2 base deltas in robot-local frame
      "delta_s_robot": 0.0,
      "delta_theta": 0.0,
      // 10 raw velocities (bonus, not strictly needed for VLA)
      "v_lift": 0.0, "v_arm": 0.0, "v_wrist_yaw": 0.0, "v_wrist_pitch": 0.0,
      "v_wrist_roll": 0.0, "v_gripper": 0.0, "v_head_pan": 0.0,
      "v_head_tilt": 0.0, "v_base_forward": 0.0, "v_base_yaw": 0.0,
      // metadata
      "leader": "robot_gamepad"
    },
    "xyz": [0,0,0],     "quats": [0,0,0,1],   // AR-marker placeholders (N/A)
    "ee_pos": [0,0,0],  "ee_rot": [[1,0,0],[0,1,0],[0,0,1]],   // FK at ETL
    "waypoints": {}
  },
  "1": { ... }
}
```

### ETL transforms (do these at training-data preparation time)

For VLA training schema (14D state, 18D action) you'll need:

**State 14D from saved 13 fields:**
- `base_x_norm = base_x − episode_start_x` (episode-frame normalize)
- `base_y_norm = base_y − episode_start_y`
- `sin(theta_norm), cos(theta_norm)` (replace base_theta with sin/cos)
- `gripper_norm = clip(gripper / 70.0, 0, 1)` (or use observed max)
- Other fields pass through unchanged

**Action 18D from saved 21 fields:**
- 6 arm absolute (lift/arm/yaw/pitch/roll/gripper_norm) — direct
- 6 arm delta = absolute − state[t]
- 2 head absolute (head_pan, head_tilt) — direct
- 2 head delta = absolute − state[t]
- 2 base delta (delta_s_robot, delta_theta) — direct
- (drop the 10 v_* extras at this stage if not used)

---

## Camera details (for reference)

| Camera | Model | FOV (color) | Mount | Bus |
|---|---|---|---|---|
| Head | Intel D435if | **55.7° H × 43.3° V** (medium-wide) | sideways → image rotated 90° CW | USB 3.0 |
| Gripper | Intel D405 | 72.8° H × 57.9° V | mounted on gripper | USB 2.0 |

**Important quirks discovered:**
1. D435if image comes out **sideways** (raw `(480, 640)` but H is the actual W IRL). robot_leader auto-rotates 90° CW to upright (`--head_rotate 90` default).
2. D405 is on USB 2.0 → bandwidth-constrained. We use 15 fps RGB-only (no depth stream) to fit comfortably; head uses 15 fps RGB + depth on USB 3.0.
3. RealSense USB pipes occasionally lock up between sessions. robot_leader calls `dev.hardware_reset()` + 5s wait at startup to recover.
4. `poll_for_frames()` may return empty for many seconds after `pipeline.start()`. We `wait_for_frames(10s)` once at warmup to prime the cache, so the very first recorded frame is never black.

---

## Known limitations

1. **Gripper executed-state-as-action**: `stretch_gripper` action records `state + v×dt` but actual gripper motion comes from `move_by(60)` (much larger displacement per frame). The recorded sequence still works as pseudo-commanded targets because state[t+1] reflects the actual motion, but it's not a clean commanded-intent signal.

2. **Depth recording slow**: `FileDataRecorder.process_depth_to_bin` loads all frames in memory + casts to float32 + lzfse-compresses single-threaded. Multi-minute episodes can take several minutes to save. Default is off (`--save_depth` opt-in).

3. **Mode-switch latency**: stretch_gamepad_teleop daemon auto-restarts on boot; first run after reboot needs `start_bridge.sh` (kills daemon) or `kill <pid>` manually.

4. **Reproducible from saved video?** RGB+state are exact. Recovering depth (when not saved) requires re-running scene with same lighting.

---

## Troubleshooting

### "Robot is not homed" / RuntimeError on bridge start
```bash
ssh hello-robot@10.100.66.215 \
  'export HELLO_FLEET_PATH=/home/hello-robot/stretch_user; \
   export HELLO_FLEET_ID=stretch-se3-3139; \
   ~/.local/bin/stretch_robot_home.py'
```

### "Port /dev/hello-pimu is busy"
Another stretch_body process is holding hardware (usually the daemon).
`start_bridge.sh` kills it automatically. Manual:
```bash
pkill -f stretch_gamepad_teleop
```

### Continuous beeping
Runstop is engaged. **Physically rotate the orange button** to release.
Verify with `pimu.status['runstop_event'] == False`.

### Black/static gripper video
RealSense D405 USB pipe stuck. robot_leader does hardware_reset on
startup; if you stopped/started rapidly, reset may not have fully
applied. Wait 30s and retry, or unplug/replug the USB cable.

### Compile / import errors
```bash
ssh hello-robot@10.100.66.215 'python3 -c "import py_compile; py_compile.compile(\"/home/hello-robot/robot_leader.py\", doraise=True); print(\"OK\")"'
```

### Replay drifts away from original trajectory
- Try `--mode target` (more robust to timing drift)
- Verify `--fps 15` matches recording
- Verify start pose was reached before replay started
- Check robot started from same physical location (base_x/y/theta in odom)

---

## Branch / commit reference

This module lives on `feat/gamepad-leader` branch in
`DarlingInTheSV/stretch_ai`.

Key commits:
- `feat/gamepad-leader` initial gamepad teleop module
- `99db10` hardware_reset + wait_for_frames warmup (camera reliability)
- `e238cc6` D405 USB 2.0 lighter config
- `7c8ccb6` 15fps both cams + head rotate + crf=20
- `0a3a0ac` head 'ahead' tilts down
- `c621736` align head poses with ai_pickup convention (-65° tilt for nav)
- `0e04522` robot_replay.py
