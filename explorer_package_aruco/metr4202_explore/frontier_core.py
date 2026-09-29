"""
frontier_core.py - ROS-free frontier detection on an occupancy grid.

Kept separate from the ROS node so it can be unit-tested and simulated
offline (see test/test_frontier_core.py and tools/sim_explore.py).

Grid conventions (same as nav_msgs/OccupancyGrid):
    -1        unknown
    0..100    occupancy probability (slam_toolbox publishes 0 or 100)
    grid[row, col], row increases with +y, col increases with +x

Pipeline
--------
1. classify cells into free / occupied / unknown
2. frontier cell = free cell with an unknown 4-neighbour
3. "passable"   = free cells at least `reach_clearance` from any wall
   "safe goal"  = free cells at least `robot_radius` from any wall
4. breadth-first search from the robot through passable cells
   -> which cells are reachable, and the path length (in cells) to each
5. group frontier cells into 8-connected clusters, drop tiny ones
6. for each cluster choose ONE goal cell that is safe and reachable
   (on the cluster itself if possible, otherwise a nearby cell that has
   line of sight to the frontier). Clusters with no such cell are
   reported as unreachable and never sent to Nav2.
"""

import math
from collections import deque

import numpy as np


# --------------------------------------------------------------------------
# small grid utilities
# --------------------------------------------------------------------------
def disk_offsets(radius):
    r = int(radius)
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r]


def dilate_disk(mask, radius):
    """Binary dilation of `mask` by a disk of `radius` cells (numpy only)."""
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy, dx in disk_offsets(radius):
        out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] |= \
            mask[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return out


def classify(grid, free_thresh=25, occ_thresh=65):
    unknown = grid < 0
    free = (grid >= 0) & (grid <= free_thresh)
    occupied = grid >= occ_thresh
    return free, occupied, unknown


def frontier_mask(free, unknown):
    """Free cells with at least one unknown 4-neighbour."""
    adj = np.zeros_like(unknown)
    adj[1:, :] |= unknown[:-1, :]
    adj[:-1, :] |= unknown[1:, :]
    adj[:, 1:] |= unknown[:, :-1]
    adj[:, :-1] |= unknown[:, 1:]
    return free & adj


def cells_near(mask, center, radius):
    """True cells of `mask` within `radius` cells of `center`, nearest first."""
    r0, c0 = int(center[0]), int(center[1])
    h, w = mask.shape
    y0, y1 = max(0, r0 - radius), min(h, r0 + radius + 1)
    x0, x1 = max(0, c0 - radius), min(w, c0 + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return []
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if ys.size == 0:
        return []
    ys = ys + y0
    xs = xs + x0
    d2 = (ys - r0) ** 2 + (xs - c0) ** 2
    keep = d2 <= radius * radius
    ys, xs, d2 = ys[keep], xs[keep], d2[keep]
    order = np.argsort(d2, kind='stable')
    return list(zip(ys[order].tolist(), xs[order].tolist()))


def line_of_sight(blocked, a, b):
    """True if the straight line a->b (endpoints excluded) avoids `blocked`."""
    (r0, c0), (r1, c1) = a, b
    n = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(1, n):
        t = k / n
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if blocked[r, c]:
            return False
    return True


def bfs_distance(passable, seed):
    """4-connected BFS path length (in cells) from `seed`; -1 = unreachable."""
    h, w = passable.shape
    n = h * w
    flat = passable.ravel().tolist()        # python lists are much faster
    dist = [-1] * n                         # than numpy element access
    s = int(seed[0]) * w + int(seed[1])
    dist[s] = 0
    q = deque([s])
    while q:
        i = q.popleft()
        nd = dist[i] + 1
        c = i % w
        if c > 0:
            j = i - 1
            if flat[j] and dist[j] < 0:
                dist[j] = nd
                q.append(j)
        if c < w - 1:
            j = i + 1
            if flat[j] and dist[j] < 0:
                dist[j] = nd
                q.append(j)
        if i >= w:
            j = i - w
            if flat[j] and dist[j] < 0:
                dist[j] = nd
                q.append(j)
        if i < n - w:
            j = i + w
            if flat[j] and dist[j] < 0:
                dist[j] = nd
                q.append(j)
    return np.array(dist, dtype=np.int32).reshape(h, w)


def cluster_cells(mask, min_size=1):
    """8-connected components of `mask`; returns list of (n, 2) int arrays."""
    ys, xs = np.nonzero(mask)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    clusters = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        cells = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in remaining:
                        remaining.remove(nb)
                        stack.append(nb)
                        cells.append(nb)
        if len(cells) >= min_size:
            clusters.append(np.array(cells, dtype=np.int32))
    return clusters


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def find_frontiers(grid, resolution, robot_rc,
                   free_thresh=25, occ_thresh=65,
                   robot_radius=0.22, reach_clearance=0.17,
                   min_cluster_cells=8, goal_search_radius=0.75,
                   seed_search_radius=0.6):
    """
    Parameters
    ----------
    grid        : (H, W) int array, OccupancyGrid values
    resolution  : metres per cell
    robot_rc    : (row, col) of the robot in the grid

    Returns
    -------
    candidates : list of dicts
        goal        (row, col) cell to send to Nav2 (safe + reachable)
        anchor      (row, col) frontier cell the goal was chosen for
        size        number of frontier cells in the cluster
        path_cells  BFS path length robot -> goal, in cells
    stats : dict with diagnostic counts
    """
    grid = np.asarray(grid)
    free, occupied, unknown = classify(grid, free_thresh, occ_thresh)
    frontier = frontier_mask(free, unknown)

    goal_r = int(math.ceil(robot_radius / resolution - 1e-9))
    reach_r = min(goal_r, int(math.floor(reach_clearance / resolution + 1e-9)))
    safe_goal = free & ~dilate_disk(occupied, goal_r)
    passable = free & ~dilate_disk(occupied, reach_r)

    stats = {'frontier_cells': int(frontier.sum()), 'clusters': 0,
             'unreachable_clusters': 0, 'seed': None}

    # robot cell may itself be inside the inflated zone: use nearest passable
    seeds = cells_near(passable, robot_rc,
                       max(1, int(round(seed_search_radius / resolution))))
    if not seeds:
        return [], stats
    seed = seeds[0]
    stats['seed'] = seed

    dist = bfs_distance(passable, seed)
    goal_ok = safe_goal & (dist >= 0)

    clusters = cluster_cells(frontier, min_cluster_cells)
    stats['clusters'] = len(clusters)
    search_r = max(1, int(round(goal_search_radius / resolution)))

    out = []
    for cells in clusters:
        cy, cx = cells[:, 0].mean(), cells[:, 1].mean()
        d2 = (cells[:, 0] - cy) ** 2 + (cells[:, 1] - cx) ** 2
        order = np.argsort(d2, kind='stable')
        anchor = (int(cells[order[0], 0]), int(cells[order[0], 1]))

        goal = None
        ok = goal_ok[cells[:, 0], cells[:, 1]]
        if ok.any():
            # a frontier cell of this cluster is itself a valid goal:
            # take the one closest to the cluster centre
            idx = order[ok[order]][0]
            goal = (int(cells[idx, 0]), int(cells[idx, 1]))
        else:
            # back off into known free space, but keep sight of the frontier
            for cand in cells_near(goal_ok, anchor, search_r):
                if line_of_sight(occupied, cand, anchor):
                    goal = cand
                    break

        if goal is None:
            stats['unreachable_clusters'] += 1
            continue

        out.append({'goal': goal, 'anchor': anchor, 'size': int(len(cells)),
                    'path_cells': int(dist[goal])})
    return out, stats


def rank_candidates(candidates, resolution, info_gain_weight=0.5,
                    min_goal_distance=0.3):
    """
    Lower cost = better.  cost = path_length - w * frontier_length (metres).
    w = 0 gives classic nearest-frontier exploration; larger w prefers big
    openings.  Path length is the BFS distance through the map, NOT the
    straight-line distance, which matters a lot in a maze.
    """
    ranked = []
    for c in candidates:
        path_m = c['path_cells'] * resolution
        if path_m < min_goal_distance:
            continue
        cost = path_m - info_gain_weight * c['size'] * resolution
        ranked.append((cost, c))
    ranked.sort(key=lambda t: t[0])
    return [c for _, c in ranked]
