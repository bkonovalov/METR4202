"""Run with:  python3 -m pytest test/  (or just python3 test/test_frontier_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metr4202_explore.frontier_core import (  # noqa: E402
    dilate_disk, find_frontiers, rank_candidates)

RES = 0.05


def room(h=60, w=60):
    g = np.full((h, w), -1, dtype=np.int16)
    return g


def test_simple_corridor_frontier():
    # known free corridor 20 cells wide, open (unknown) at the right end
    g = room()
    g[10:50, 5:40] = 100            # walls block ...
    g[20:40, 5:40] = 0              # ... free corridor
    g[20:40, 5] = 100               # closed at the left end
    cands, stats = find_frontiers(g, RES, (30, 12))
    assert len(cands) >= 1
    goal = cands[0]['goal']
    # goal must be on/near the open end and away from the side walls
    assert goal[1] >= 30
    assert 24 <= goal[0] <= 35


def test_goal_never_hugs_wall():
    g = room()
    g[20:40, 5:40] = 0
    g[19, 5:40] = 100
    g[40, 5:40] = 100
    cands, _ = find_frontiers(g, RES, (30, 10), robot_radius=0.22)
    near_wall = dilate_disk(g == 100, 5)
    for c in cands:
        assert not near_wall[c['goal']]


def test_unreachable_pocket_is_rejected():
    g = room()
    g[:, :] = 100
    g[20:40, 5:25] = 0               # robot's room, fully enclosed
    g[20:40, 30:50] = 0              # a separate pocket behind a wall...
    g[20:40, 50:55] = -1             # ...that touches unknown space
    cands, stats = find_frontiers(g, RES, (30, 10))
    assert cands == []
    assert stats['unreachable_clusters'] == 1


def test_ranking_prefers_near():
    cands = [{'goal': (0, 0), 'anchor': (0, 0), 'size': 10, 'path_cells': 200},
             {'goal': (1, 1), 'anchor': (1, 1), 'size': 10, 'path_cells': 40}]
    ranked = rank_candidates(cands, RES, info_gain_weight=0.0)
    assert ranked[0]['path_cells'] == 40


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
