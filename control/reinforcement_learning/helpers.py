import os
import time
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt


@contextmanager
def timing(label, timings):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    if label in timings:
        timings[label] += elapsed
    else:
        timings[label] = elapsed


def plot_combined(
    run_dir: str,
    timings: dict,
    indexes_to_plot: np.ndarray,
    scores: list[float],
    env_name: str,
    agent_type: str,
):
    # Prepare data for the pie chart
    labals_in_minutes = [f"{key} ({value / 60:.2f}m)" for key, value in timings.items()]
    sizes = list(timings.values())

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # Pie chart on the first subplot
    wedges, texts, autotexts = ax1.pie(
        sizes, autopct="%1.1f%%", shadow=True, startangle=90
    )
    ax1.axis("equal")  # Ensure it's a circle
    ax1.set_title(
        f"Timing Breakdown up to E {indexes_to_plot[-1]}, Total Time: {sum(sizes) / 60:.2f}m"
    )
    ax1.legend(wedges, labals_in_minutes, title="Timings", loc="upper right")

    # Line plot on the second subplot
    ax2.plot(indexes_to_plot, scores, label="Perf. Score")
    ax2.set_title(f"Performance Score for {agent_type} on {env_name}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Rolling Score")
    ax2.legend()

    # Save and show the figure
    figure_file = os.path.join(run_dir, "combined_plot.png")
    plt.savefig(figure_file)
    plt.show()


def plot_episode_timing(episode_index, timings):
    labels = []
    sizes = []
    for key, value in timings.items():
        labels.append(f"{key} ({value:.2f}s)")
        sizes.append(value)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, autopct="%1.1f%%", shadow=True, startangle=90
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(
        f"Timing Breakdown up to Episode {episode_index - 1}, Total Time: {sum(sizes):.2f}s"
    )
    plt.legend(wedges, labels, title="Timings")
    plt.show()


def plot_learning_curve(x, scores, run_dir):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    figure_file = os.path.join(run_dir, "learning_curve.png")
    plt.savefig(figure_file)
    plt.show()