#!/usr/bin/env python3
import os
import time
import math
import csv
import signal
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


# ============================================================
# CONFIG
# ============================================================

# Constant particle count for B2
N_PARTICLES = 1000

# Motion noise scaling (5 runs)
# Multiplies base alphas (alpha1..alpha4)
NOISE_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]

BASE_ALPHAS = {
    "alpha1": 0.05,
    "alpha2": 0.05,
    "alpha3": 0.10,
    "alpha4": 0.05,
}

# Topics
TOPIC_GT = "/robot_gt"
TOPIC_EST = "/robot_estimated_odometry"

# Timing
SIM_DURATION_SEC = 35.0
WARMUP_SEC = 2.0
SPIN_HZ = 50.0

# Convergence definition
CONV_POS_THRESH_M = 0.35
CONV_HOLD_SEC = 2.0

# Stability window (last seconds of run, after warmup)
STABILITY_WINDOW_SEC = 10.0

# Workspace + setup
WS_DIR = Path(__file__).resolve().parents[1]
SETUP_CMD = f"source {WS_DIR}/install/setup.zsh"

# Commands
FAKE_CMD = f"{SETUP_CMD} && ros2 launch fake_robot fake_robot.launch.py"
MCL_CMD_BASE = f"{SETUP_CMD} && ros2 run mcl_localization mcl_node --ros-args --log-level INFO"

# Output
OUT_DIR = WS_DIR / "analyze" / "B2"
LOG_DIR = OUT_DIR / "runtime_logs"
OUT_CSV = OUT_DIR / "B2_results.csv"
PLOT_RMSE = OUT_DIR / "B2_rmse_vs_motion_noise.png"
PLOT_STAB = OUT_DIR / "B2_stability_vs_motion_noise.png"
PLOT_ERR_TIME = OUT_DIR / "B2_pos_error_over_time_all_scales.png"


# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def odom_to_pose(msg: Odometry) -> Tuple[float, float, float]:
    x = float(msg.pose.pose.position.x)
    y = float(msg.pose.pose.position.y)
    th = yaw_from_quat(msg.pose.pose.orientation)
    return x, y, th

def stop_ros_process(proc: Optional[subprocess.Popen], name: str, timeout_sec: float = 12.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    print(f"[INFO] Stopping {name} (SIGINT)")
    try:
        os.killpg(pgid, signal.SIGINT)
    except Exception:
        return

    try:
        proc.wait(timeout=timeout_sec)
        return
    except subprocess.TimeoutExpired:
        print(f"[WARN] {name} did not stop after SIGINT, sending SIGTERM")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass

    try:
        proc.wait(timeout=4)
        return
    except subprocess.TimeoutExpired:
        print(f"[WARN] {name} did not stop after SIGTERM, killing it")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


# ============================================================
# Evaluator
# ============================================================

class Evaluator(Node):
    def __init__(self):
        super().__init__("b2_evaluator")

        self.gt: Optional[Odometry] = None
        self.est: Optional[Odometry] = None

        self.create_subscription(Odometry, TOPIC_GT, self._cb_gt, 50)
        self.create_subscription(Odometry, TOPIC_EST, self._cb_est, 50)

        self.t0_wall: Optional[float] = None
        self.times: List[float] = []
        self.pos_errs: List[float] = []
        self.theta_errs: List[float] = []

        # For trace CSV (GT + EST)
        self.gt_xs: List[float] = []
        self.gt_ys: List[float] = []
        self.gt_ths: List[float] = []
        self.est_xs: List[float] = []
        self.est_ys: List[float] = []
        self.est_ths: List[float] = []

    def _cb_gt(self, msg: Odometry):
        self.gt = msg

    def _cb_est(self, msg: Odometry):
        self.est = msg

    def reset(self):
        self.gt = None
        self.est = None
        self.t0_wall = None

        self.times.clear()
        self.pos_errs.clear()
        self.theta_errs.clear()

        self.gt_xs.clear(); self.gt_ys.clear(); self.gt_ths.clear()
        self.est_xs.clear(); self.est_ys.clear(); self.est_ths.clear()

    def step(self):
        if self.gt is None or self.est is None:
            return

        now = time.time()
        if self.t0_wall is None:
            self.t0_wall = now
        t = now - self.t0_wall

        gx, gy, gth = odom_to_pose(self.gt)
        ex, ey, eth = odom_to_pose(self.est)

        dpos = math.hypot(ex - gx, ey - gy)
        dth = abs(wrap_to_pi(eth - gth))

        self.times.append(t)
        self.pos_errs.append(dpos)
        self.theta_errs.append(dth)

        self.gt_xs.append(gx); self.gt_ys.append(gy); self.gt_ths.append(gth)
        self.est_xs.append(ex); self.est_ys.append(ey); self.est_ths.append(eth)

    def compute_metrics(self):
        # warmup cut
        pairs = [(t, pe, te) for (t, pe, te) in zip(self.times, self.pos_errs, self.theta_errs) if t >= WARMUP_SEC]
        if len(pairs) < 10:
            return {
                "samples": len(pairs),
                "rmse_pos": float("nan"),
                "rmse_theta": float("nan"),
                "mean_pos": float("nan"),
                "conv_time": None,
                "stability_pos_std": float("nan"),
            }

        ts = [p[0] for p in pairs]
        pes = [p[1] for p in pairs]
        tes = [p[2] for p in pairs]

        rmse_pos = math.sqrt(sum(pe * pe for pe in pes) / len(pes))
        rmse_theta = math.sqrt(sum(te * te for te in tes) / len(tes))
        mean_pos = sum(pes) / len(pes)

        # convergence
        conv_time = None
        for i in range(len(ts)):
            t_i = ts[i]
            t_end = t_i + CONV_HOLD_SEC
            j = i
            while j < len(ts) and ts[j] <= t_end:
                j += 1
            if j <= i + 2:
                continue
            if all(pe < CONV_POS_THRESH_M for pe in pes[i:j]):
                conv_time = t_i
                break

        # stability: std of pos error in last STABILITY_WINDOW_SEC (after warmup)
        t_last = ts[-1]
        t_from = max(WARMUP_SEC, t_last - STABILITY_WINDOW_SEC)
        window = [pe for (t, pe) in zip(ts, pes) if t >= t_from]
        if len(window) >= 5:
            m = sum(window) / len(window)
            stability_std = math.sqrt(sum((x - m) ** 2 for x in window) / len(window))
        else:
            stability_std = float("nan")

        return {
            "samples": len(pairs),
            "rmse_pos": rmse_pos,
            "rmse_theta": rmse_theta,
            "mean_pos": mean_pos,
            "conv_time": conv_time,
            "stability_pos_std": stability_std,
        }


def write_trace_csv(node: Evaluator, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gt_x", "gt_y", "gt_theta", "est_x", "est_y", "est_theta", "pos_err", "theta_err"])
        for i in range(len(node.times)):
            w.writerow([
                node.times[i],
                node.gt_xs[i], node.gt_ys[i], node.gt_ths[i],
                node.est_xs[i], node.est_ys[i], node.est_ths[i],
                node.pos_errs[i], node.theta_errs[i],
            ])


# ============================================================
# Run one experiment
# ============================================================

def run_one(scale: float, run_idx: int):
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)

    a1 = BASE_ALPHAS["alpha1"] * scale
    a2 = BASE_ALPHAS["alpha2"] * scale
    a3 = BASE_ALPHAS["alpha3"] * scale
    a4 = BASE_ALPHAS["alpha4"] * scale

    run_dir = LOG_DIR / f"run_{run_idx:02d}_scale{scale}"
    ensure_dir(run_dir)

    fake_log = run_dir / "fake_robot.log"
    mcl_log = run_dir / "mcl_node.log"
    trace_path = run_dir / "trace.csv"

    print("\n================================================")
    print(f"[B2] Run {run_idx}/5 | scale={scale} | N={N_PARTICLES}")
    print(f"Alphas: ({a1:.4f}, {a2:.4f}, {a3:.4f}, {a4:.4f})")
    print(f"Trace:  {trace_path}")
    print("================================================")

    fake_proc = subprocess.Popen(
        FAKE_CMD,
        shell=True,
        executable="/bin/zsh",
        start_new_session=True,
        stdout=open(fake_log, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2.0)

    mcl_cmd = (
        f"{MCL_CMD_BASE} "
        f"-p number_of_particles:={N_PARTICLES} "
        f"-p alpha1:={a1} -p alpha2:={a2} -p alpha3:={a3} -p alpha4:={a4}"
    )
    mcl_proc = subprocess.Popen(
        mcl_cmd,
        shell=True,
        executable="/bin/zsh",
        start_new_session=True,
        stdout=open(mcl_log, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )

    rclpy.init()
    node = Evaluator()
    node.reset()

    # wait for first messages
    t_wait = time.time()
    while time.time() - t_wait < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        node.step()
        if node.gt is not None and node.est is not None:
            break

    t_start = time.time()
    dt = 1.0 / SPIN_HZ
    while time.time() - t_start < SIM_DURATION_SEC:
        rclpy.spin_once(node, timeout_sec=0.0)
        node.step()
        time.sleep(dt)

    m = node.compute_metrics()

    # write trace for multi-curve plot
    write_trace_csv(node, trace_path)

    node.destroy_node()
    rclpy.shutdown()

    stop_ros_process(mcl_proc, "mcl_node")
    stop_ros_process(fake_proc, "fake_robot")

    print(f"[B2] Done scale={scale} | rmse_pos={m['rmse_pos']:.4f} | stab_std={m['stability_pos_std']:.4f} | conv={m['conv_time']}")

    return {
        "task": "B2",
        "run_idx": run_idx,
        "number_of_particles": N_PARTICLES,
        "noise_scale": scale,
        "alpha1": a1,
        "alpha2": a2,
        "alpha3": a3,
        "alpha4": a4,
        "duration_sec": SIM_DURATION_SEC,
        "warmup_sec": WARMUP_SEC,
        "samples": m["samples"],
        "rmse_pos_m": m["rmse_pos"],
        "rmse_theta_rad": m["rmse_theta"],
        "mean_pos_err_m": m["mean_pos"],
        "conv_time_sec": m["conv_time"],
        "stability_pos_std_m": m["stability_pos_std"],
        "stability_window_sec": STABILITY_WINDOW_SEC,
        "fake_robot_log": str(fake_log),
        "mcl_node_log": str(mcl_log),
        "trace_csv": str(trace_path),
    }


# ============================================================
# Output
# ============================================================

def write_csv(rows: list, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def plot_xy(rows: list, x_key: str, y_key: str, title: str, out_path: Path, x_label: str, y_label: str):
    import matplotlib.pyplot as plt
    xs = [r[x_key] for r in rows]
    ys = [r[y_key] for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    ensure_dir(out_path.parent)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_pos_error_over_time(rows: list, out_path: Path):
    import matplotlib.pyplot as plt
    ensure_dir(out_path.parent)
    plt.figure()

    for r in rows:
        trace = Path(r["trace_csv"])
        if not trace.exists():
            continue

        ts, errs = [], []
        with open(trace, "r") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                t = float(row["t"])
                if t < WARMUP_SEC:
                    continue
                ts.append(t)
                errs.append(float(row["pos_err"]))

        plt.plot(ts, errs, label=f"scale={r['noise_scale']}")

    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.title("B2: Position error over time (all motion noise scales)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("===== STARTING B2: Motion Noise Variation =====")
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)

    results = []
    for idx, s in enumerate(NOISE_SCALES, start=1):
        results.append(run_one(s, idx))

    write_csv(results, OUT_CSV)

    plot_xy(results, "noise_scale", "rmse_pos_m",
            "B2: RMSE vs motion noise scale", PLOT_RMSE,
            "motion noise scale (alphas)", "RMSE position [m]")

    plot_xy(results, "noise_scale", "stability_pos_std_m",
            "B2: Stability vs motion noise scale", PLOT_STAB,
            "motion noise scale (alphas)", "Std(position error) [m] (last window)")

    plot_pos_error_over_time(results, PLOT_ERR_TIME)

    print("===== B2 COMPLETED =====")
    print(f"CSV:  {OUT_CSV}")
    print(f"Plot: {PLOT_RMSE}")
    print(f"Plot: {PLOT_STAB}")
    print(f"Plot: {PLOT_ERR_TIME}")
