"""Generate a linearly partitioned grid (square split into random convex polytopes)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class Line:
    normal: np.ndarray  # shape (2,)
    offset: float  # n . x = offset


def _clip_polygon_halfspace(poly: Polygon, line: Line, keep_positive: bool) -> Polygon:
    """Clip polygon against a line halfspace.

    keep_positive keeps points where n·x >= offset (within eps).
    keep_positive=False keeps points where n·x <= offset.
    """
    if not poly:
        return []

    n = line.normal
    c = line.offset
    eps = 1e-9

    def side(p: Point) -> float:
        return float(n[0] * p[0] + n[1] * p[1] - c)

    def inside(val: float) -> bool:
        return val >= -eps if keep_positive else val <= eps

    def intersect(p1: Point, p2: Point, v1: float, v2: float) -> Point:
        t = v1 / (v1 - v2 + 1e-12)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    output: Polygon = []
    prev = poly[-1]
    prev_val = side(prev)
    prev_inside = inside(prev_val)

    for curr in poly:
        curr_val = side(curr)
        curr_inside = inside(curr_val)

        if curr_inside:
            if not prev_inside:
                output.append(intersect(prev, curr, prev_val, curr_val))
            output.append(curr)
        elif prev_inside:
            output.append(intersect(prev, curr, prev_val, curr_val))

        prev, prev_val, prev_inside = curr, curr_val, curr_inside

    return output


def _polygon_area(poly: Polygon) -> float:
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(poly: Polygon) -> Point:
    if len(poly) < 3:
        return (0.0, 0.0)
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-12:
        return (float(np.mean(x)), float(np.mean(y)))
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area))
    return (cx, cy)


def _edge_midpoint(poly: Polygon, edge_idx: int) -> Point:
    p1 = poly[edge_idx]
    p2 = poly[(edge_idx + 1) % len(poly)]
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _split_polygon_by_line(poly: Polygon, line: Line) -> Tuple[Polygon, Polygon]:
    pos = _clip_polygon_halfspace(poly, line, keep_positive=True)
    neg = _clip_polygon_halfspace(poly, line, keep_positive=False)
    return pos, neg


def generate_partition(
    num_lines: int = 10,
    square_min: float = 0.0,
    square_max: float = 1.0,
    seed: int | None = 0,
) -> List[Polygon]:
    rng = np.random.default_rng(seed)

    square: Polygon = [
        (square_min, square_min),
        (square_max, square_min),
        (square_max, square_max),
        (square_min, square_max),
    ]

    polygons: List[Polygon] = [square]

    for _ in range(num_lines):
        theta = rng.uniform(0, 2 * np.pi)
        normal = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        offset = rng.uniform(square_min, square_max) * (normal[0] + normal[1])
        line = Line(normal=normal, offset=offset)

        next_polys: List[Polygon] = []
        for poly in polygons:
            pos, neg = _split_polygon_by_line(poly, line)
            if abs(_polygon_area(pos)) > 1e-6:
                next_polys.append(pos)
            if abs(_polygon_area(neg)) > 1e-6:
                next_polys.append(neg)
        polygons = next_polys

    return polygons


def plot_partition(polygons: Iterable[Polygon], save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    polygons = list(polygons)
    for poly in polygons:
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1)
        ax.fill(xs, ys, alpha=0.08)

    if polygons:
        target = max(polygons, key=lambda p: abs(_polygon_area(p)))
        centroid = _polygon_centroid(target)
        edge_mid = _edge_midpoint(target, 0)
        ax_dir = (centroid[0] - edge_mid[0], centroid[1] - edge_mid[1])
        point_a = (edge_mid[0] + 0.08 * ax_dir[0], edge_mid[1] + 0.08 * ax_dir[1])
        point_b = centroid

        ax.scatter([point_a[0], point_b[0]], [point_a[1], point_b[1]], color="red", zorder=5)
        ax.text(point_a[0], point_a[1], "A", color="red", fontsize=12, ha="left", va="bottom")
        ax.text(point_b[0], point_b[1], "B", color="red", fontsize=12, ha="left", va="bottom")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Intuitive visual explanation")
    ax.grid(False)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a linearly partitioned grid.")
    parser.add_argument("--num-lines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    polys = generate_partition(num_lines=args.num_lines, seed=args.seed)
    plot_partition(polys, save_path=args.save)


if __name__ == "__main__":
    main()
