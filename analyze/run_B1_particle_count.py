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
# CONFIG (anpassen falls nötig)
# ============================================================

# Particle counts (5 runs)
PARTICLE_COUNTS = [100, 500, 1000, 2000, 5000]

# Topics for evaluation
TOPIC_GT = "/robot_gt"
TOPIC_EST = "/robot_estimated_odometry"

# Timing
SIM_DURATION_SEC = 35.0
WARMUP_SEC = 2.0
SPIN_HZ = 50.0

# Convergence definition (position error)
CONV_POS_THRESH_M = 0.35     # meters
CONV_HOLD_SEC = 2.0          # must stay below threshold for this long

# Workspace + setup
# Assumption: script is in <ws>/analyze/
WS_DIR = Path(__file__).resolve().parents[1]
SETUP_CMD = f"source {WS_DIR}/install/setup.zsh"

# Commands
FAKE_CMD = f"{SETUP_CMD} && ros2 launch fake_robot fake_robot.launch.py"
MCL_CMD_BASE = f"{SETUP_CMD} && ros2 run mcl_localization mcl_node --ros-args --log-level INFO"

# Output
OUT_DIR = WS_DIR / "analyze" / "B1"
LOG_DIR = OUT_DIR / "runtime_logs"
OUT_CSV = OUT_DIR / "B1_results.csv"
PLOT_PATH = OUT_DIR / "B1_rmse_vs_particles.png"
PLOT_ERR_TIME = OUT_DIR / "B1_pos_error_over_time_all_particles.png"


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
    """Gracefully stop a process group started with start_new_session=True."""
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
# Evaluator Node
# ============================================================

class Evaluator(Node):
    def __init__(self):
        super().__init__("b1_evaluator")
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
        # apply warmup
        pairs = [(t, pe, te) for (t, pe, te) in zip(self.times, self.pos_errs, self.theta_errs) if t >= WARMUP_SEC]
        if len(pairs) < 10:
            return {
                "samples": len(pairs),
                "rmse_pos": float("nan"),
                "rmse_theta": float("nan"),
                "mean_pos": float("nan"),
                "mean_theta": float("nan"),
                "conv_time": None,
            }

        ts = [p[0] for p in pairs]
        pes = [p[1] for p in pairs]
        tes = [p[2] for p in pairs]

        rmse_pos = math.sqrt(sum(pe * pe for pe in pes) / len(pes))
        rmse_theta = math.sqrt(sum(te * te for te in tes) / len(tes))
        mean_pos = sum(pes) / len(pes)
        mean_theta = sum(tes) / len(tes)

        # convergence time: first time pos_err < thresh for hold window
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

        return {
            "samples": len(pairs),
            "rmse_pos": rmse_pos,
            "rmse_theta": rmse_theta,
            "mean_pos": mean_pos,
            "mean_theta": mean_theta,
            "conv_time": conv_time,
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
# Single run
# ============================================================

def run_one(particle_count: int, run_idx: int):
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)

    run_dir = LOG_DIR / f"run_{run_idx:02d}_N{particle_count}"
    ensure_dir(run_dir)

    fake_log = run_dir / "fake_robot.log"
    mcl_log = run_dir / "mcl_node.log"
    trace_path = run_dir / "trace.csv"

    print("\n================================================")
    print(f"[B1] Run {run_idx}/5 | N={particle_count}")
    print(f"Logs:  {run_dir}")
    print(f"Trace: {trace_path}")
    print("================================================")

    # Start fake_robot
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

    # Start mcl_node with particle count
    mcl_cmd = f"{MCL_CMD_BASE} -p number_of_particles:={particle_count}"
    mcl_proc = subprocess.Popen(
        mcl_cmd,
        shell=True,
        executable="/bin/zsh",
        start_new_session=True,
        stdout=open(mcl_log, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Evaluate
    rclpy.init()
    node = Evaluator()
    node.reset()

    # small wait for first messages (max 5s)
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

    metrics = node.compute_metrics()

    # write trace for multi-curve plot
    write_trace_csv(node, trace_path)

    node.destroy_node()
    rclpy.shutdown()

    # Stop processes
    stop_ros_process(mcl_proc, "mcl_node")
    stop_ros_process(fake_proc, "fake_robot")

    print(f"[B1] Done N={particle_count} | rmse_pos={metrics['rmse_pos']:.4f} | conv={metrics['conv_time']}")

    return {
        "task": "B1",
        "run_idx": run_idx,
        "number_of_particles": particle_count,
        "duration_sec": SIM_DURATION_SEC,
        "warmup_sec": WARMUP_SEC,
        "samples": metrics["samples"],
        "rmse_pos_m": metrics["rmse_pos"],
        "rmse_theta_rad": metrics["rmse_theta"],
        "mean_pos_err_m": metrics["mean_pos"],
        "mean_theta_err_rad": metrics["mean_theta"],
        "conv_time_sec": metrics["conv_time"],
        "conv_thresh_m": CONV_POS_THRESH_M,
        "conv_hold_sec": CONV_HOLD_SEC,
        "fake_robot_log": str(fake_log),
        "mcl_node_log": str(mcl_log),
        "trace_csv": str(trace_path),
    }


# ============================================================
# Main
# ============================================================

def write_results_csv(rows: list, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def plot_rmse(rows: list, plot_path: Path):
    import matplotlib.pyplot as plt

    xs = [r["number_of_particles"] for r in rows]
    ys = [r["rmse_pos_m"] for r in rows]

    # sort by x
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    ensure_dir(plot_path.parent)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    # plt.xscale("log")  # optional
    plt.xlabel("number_of_particles")
    plt.ylabel("RMSE position [m]")
    plt.title("B1: RMSE vs particle count")
    plt.grid(True)
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()

def plot_pos_error_over_time(rows: list, plot_path: Path):
    import matplotlib.pyplot as plt

    ensure_dir(plot_path.parent)
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

        plt.plot(ts, errs, label=f"N={r['number_of_particles']}")

    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.title("B1: Position error over time (all particle counts)")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("===== STARTING B1: Particle Count Variation =====")
    ensure_dir(OUT_DIR)
    ensure_dir(LOG_DIR)

    results = []
    for idx, N in enumerate(PARTICLE_COUNTS, start=1):
        results.append(run_one(N, idx))

    write_results_csv(results, OUT_CSV)
    plot_rmse(results, PLOT_PATH)
    plot_pos_error_over_time(results, PLOT_ERR_TIME)

    print("===== B1 COMPLETED =====")
    print(f"CSV:  {OUT_CSV}")
    print(f"Plot: {PLOT_PATH}")
    print(f"Plot: {PLOT_ERR_TIME}")
