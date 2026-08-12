#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import lpips
import torch

from lpips_utils import calculate_lpips_distance, load_rgb_image, pair_images


def parse_method(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Methods must use NAME=IMAGE_DIRECTORY.")
    name, directory = value.split("=", 1)
    if not name.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("Both method name and image directory are required.")
    return name.strip(), Path(directory)


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "method"


def read_previous_ranking(path):
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8-sig") as csvfile:
        rows = list(csv.DictReader(csvfile))
    required = {"method", "average_lpips"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Previous summary must contain columns {sorted(required)}: {path}"
        )
    return [
        row["method"]
        for row in sorted(rows, key=lambda row: float(row["average_lpips"]))
    ]


def evaluate_method(loss_fn, device, reference_dir, method_name, method_dir, image_size):
    rows = []
    for image_id, reference_path, reconstruction_path in pair_images(
        reference_dir, method_dir
    ):
        reference = load_rgb_image(reference_path, image_size)
        reconstruction = load_rgb_image(reconstruction_path, image_size)
        distance = calculate_lpips_distance(
            loss_fn, reference, reconstruction, device
        )
        rows.append(
            {
                "image_id": image_id,
                "reference": str(reference_path),
                "reconstruction": str(reconstruction_path),
                "lpips": distance,
            }
        )

    average = sum(row["lpips"] for row in rows) / len(rows)
    return rows, {
        "method": method_name,
        "average_lpips": average,
        "num_pairs": len(rows),
        "network": "alex",
        "input_range": "[-1,1]",
        "image_size": image_size if image_size is not None else "original",
    }


def write_method_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["image_id", "reference", "reconstruction", "lpips"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "lpips": f"{row['lpips']:.8f}"})


def write_summary(path, summaries):
    fields = [
        "rank",
        "method",
        "average_lpips",
        "num_pairs",
        "network",
        "input_range",
        "image_size",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for rank, summary in enumerate(summaries, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **summary,
                    "average_lpips": f"{summary['average_lpips']:.8f}",
                }
            )


def write_ranking_report(path, summaries, previous_ranking):
    current_ranking = [summary["method"] for summary in summaries]
    lines = [
        "# LPIPS ranking",
        "",
        "LPIPS uses AlexNet with standard `[-1,1]` inputs. Lower is better.",
        "",
    ]
    for rank, summary in enumerate(summaries, start=1):
        lines.append(
            f"{rank}. {summary['method']}: {summary['average_lpips']:.8f} "
            f"({summary['num_pairs']} pairs)"
        )

    lines.extend(["", "## Ranking change", ""])
    if previous_ranking is None:
        lines.append("No previous summary was supplied; ranking change was not evaluated.")
    else:
        shared_methods = set(current_ranking) & set(previous_ranking)
        previous_shared = [name for name in previous_ranking if name in shared_methods]
        current_shared = [name for name in current_ranking if name in shared_methods]
        if previous_shared == current_shared:
            lines.append("The relative order of shared methods is unchanged.")
        else:
            lines.append(
                "The relative order changed: "
                f"`{' > '.join(previous_shared)}` -> `{' > '.join(current_shared)}`."
            )
        missing = sorted(set(previous_ranking) - set(current_ranking))
        added = sorted(set(current_ranking) - set(previous_ranking))
        if missing:
            lines.append(f"Methods absent from the new run: {', '.join(missing)}.")
        if added:
            lines.append(f"Methods added in the new run: {', '.join(added)}.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate standard LPIPS for multiple reconstruction methods."
    )
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument(
        "--method",
        required=True,
        action="append",
        type=parse_method,
        metavar="NAME=DIR",
        help="Method name and reconstruction directory; repeat for multiple methods.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("lpips_results"))
    parser.add_argument(
        "--previous-summary",
        type=Path,
        help="Optional prior summary CSV with method and average_lpips columns.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square evaluation size; use 0 to preserve original size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    args = parser.parse_args()

    if args.image_size < 0:
        parser.error("--image-size must be zero or a positive integer.")
    image_size = args.image_size or None
    method_names = [name for name, _ in args.method]
    if len(method_names) != len(set(method_names)):
        parser.error("Method names must be unique.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    summaries = []
    for method_name, method_dir in args.method:
        rows, summary = evaluate_method(
            loss_fn,
            device,
            args.original_dir,
            method_name,
            method_dir,
            image_size,
        )
        write_method_rows(
            args.output_dir / f"lpips_{safe_filename(method_name)}.csv", rows
        )
        summaries.append(summary)
        print(
            f"{method_name}: LPIPS={summary['average_lpips']:.8f} "
            f"({summary['num_pairs']} pairs)"
        )

    summaries.sort(key=lambda item: item["average_lpips"])
    write_summary(args.output_dir / "lpips_summary.csv", summaries)
    previous_ranking = read_previous_ranking(args.previous_summary)
    write_ranking_report(
        args.output_dir / "lpips_ranking.md", summaries, previous_ranking
    )
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
