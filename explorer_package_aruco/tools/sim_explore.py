#!/usr/bin/env python3
"""
Offline sanity check for the frontier logic - no ROS, no Gazebo.

Builds a random maze, gives a fake robot a 360-degree "lidar" with the
TurtleBot3's 3.5 m range, and repeatedly:
    detect frontiers -> pick the best -> teleport there -> scan
until no reachable frontier is left.  Prints coverage and saves a PNG.

    python3 tools/sim_explore.py            # default 6x6 maze
    python3 tools/sim_explore.py --seed 3 --size 8 --weight 0
"""
import argparse
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metr4202_explore.frontier_core import (  # noqa: E402
    find_frontiers, rank_candidates)

RES = 0.05


def make_maze(n, cell=16, wall=2, loops=0.1, seed=0):
    """True = wall.  cell=16 -> 0.8 m pitch, 0.7 m corridors."""
    rng = random.Random(seed)
    H = W = n * cell + wall
    occ = np.ones((H, W), bool)
    for j in range(n):
        for i in range(n):
            occ[j * cell + wall:(j + 1) * cell, i * cell + wall:(i + 1) * cell] = False

    def open_between(a, b):
        (i, j), (k, l) = a, b
        if k == i + 1:
            occ[j * cell + wall:(j + 1) * cell, (i + 1) * cell:(i + 1) * cell + wall] = False
        elif k == i - 1:
            open_between(b, a)
        elif l == j + 1:
            occ[(j + 1) * cell:(j + 1) * cell + wall, i * cell + wall:(i + 1) * cell] = False
        elif l == j - 1:
            open_between(b, a)

    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= i + di < n and 0 <= j + dj < n and (i + di, j + dj) not in seen]
        if not nbrs:
            stack.pop()
            continue
        nxt = rng.choice(nbrs)
        open_between((i, j), nxt)
        seen.add(nxt)
        stack.append(nxt)
    # knock out a few extra walls so the maze has loops
    for j in range(n):
        for i in range(n - 1):
            if rng.random() < loops:
                open_between((i, j), (i + 1, j))
    return occ


def scan(truth, known, rc, max_range=3.5, n_rays=360):
    r0, c0 = rc
    h, w = truth.shape
    steps = int(max_range / RES * 2)
    for k in range(n_rays):
        a = 2 * math.pi * k / n_rays
        sa, ca = math.sin(a), math.cos(a)
        for s in range(steps + 1):
            r = int(round(r0 + 0.5 * s * sa))
            c = int(round(c0 + 0.5 * s * ca))
            if not (0 <= r < h and 0 <= c < w):
                break
            if truth[r, c]:
                known[r, c] = 100
                break
            known[r, c] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--weight', type=float, default=0.5, help='info_gain_weight')
    ap.add_argument('--png', default='sim_explore.png')
    args = ap.parse_args()

    truth = make_maze(args.size, seed=args.seed)
    known = np.full(truth.shape, -1, dtype=np.int16)
    robot = (9, 9)                       # centre of maze cell (0, 0)
    scan(truth, known, robot)

    trail = [robot]
    blacklist = []
    travelled = 0.0
    for it in range(500):
        cands, stats = find_frontiers(known, RES, robot)
        ranked = [c for c in rank_candidates(cands, RES, args.weight)
                  if all(math.dist(c['goal'], b) * RES > 0.5 for b in blacklist)]
        if not ranked:
            print(f'done after {it} goals: {stats}')
            break
        best = ranked[0]
        travelled += best['path_cells'] * RES
        robot = best['goal']
        blacklist.append(robot)          # "visited" - never resend
        trail.append(robot)
        scan(truth, known, robot)

    free_truth = ~truth
    seen_free = (known == 0) & free_truth
    cov = seen_free.sum() / free_truth.sum()
    print(f'goals={len(trail) - 1}  path~{travelled:.1f} m  map coverage={cov:.1%}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        img = np.full(truth.shape + (3,), 0.55)
        img[known == 0] = 1.0
        img[known == 100] = 0.0
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, origin='lower')
        t = np.array(trail)
        ax.plot(t[:, 1], t[:, 0], '-o', ms=3, lw=1, color='tab:red')
        ax.set_title(f'{len(trail) - 1} goals, coverage {cov:.1%}')
        fig.savefig(args.png, dpi=120, bbox_inches='tight')
        print('saved', args.png)
    except ImportError:
        pass
    return cov


if __name__ == '__main__':
    main()
