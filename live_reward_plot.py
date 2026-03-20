import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import pandas as pd


def latest_reward_csv(log_dir: str) -> str:
    pattern = os.path.join(log_dir, "reward_trace_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No reward csv found in {log_dir}")
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Live reward curve from training CSV")
    parser.add_argument("--csv", type=str, default="", help="Path to reward CSV; default is newest in logs/")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing reward CSV files")
    parser.add_argument("--refresh-sec", type=float, default=1.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else latest_reward_csv(args.log_dir)
    print(f"[LivePlot] watching: {csv_path}")

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    while True:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                time.sleep(args.refresh_sec)
                continue

            ax.clear()
            ax.plot(
                df["step"],
                df["reward_step"],
                color="#7f8c8d",
                alpha=0.35,
                linewidth=1.0,
                label="reward_step",
            )
            ax.plot(
                df["step"],
                df["reward_step_ma200"],
                color="#e74c3c",
                linewidth=2.2,
                label="reward_step_ma200",
            )

            ep_end = df[df["done"] == 1]
            if not ep_end.empty:
                ax.scatter(
                    ep_end["step"],
                    ep_end["episode_reward_running"],
                    s=18,
                    color="#2ecc71",
                    alpha=0.8,
                    label="episode_end_reward",
                )

            ax.set_title("Wukong Training Reward (Live)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()

            plt.pause(args.refresh_sec)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[LivePlot] read failed: {exc}")
            time.sleep(args.refresh_sec)


if __name__ == "__main__":
    main()
