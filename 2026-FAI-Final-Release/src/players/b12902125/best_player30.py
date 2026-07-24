"""
BestPlayer30 -- a real ALPHAZERO-STYLE policy: a deeper network over a NEAR-RAW
board/hand encoding that LEARNS its own features, deployed WITHOUT test-time
search (microsecond numpy forward, no rollout / no opponent inference).

Why this differs from the BP19 learned-policy attempt (which capped at ~BP19):
  BP19/the first BP30 scored cards with 18 HAND-CRAFTED scalars through a tiny
  24-unit MLP. That representation was the ceiling: even fed clean,
  search-improved targets, the net could not fit a better policy (distill loss
  stayed high, weights barely moved). The public leaderboard's `noob_alpha_policy`
  beats its `_search` siblings precisely because it learns from a rich raw
  encoding with a larger net, not from a handful of engineered numbers.

BP30 therefore feeds:
  * GLOBAL (shared across candidates): raw multi-hot of my hand (104) and of the
    still-live unseen cards (104), the four row ends/lengths/penalties, score
    situation, and round progress.
  * PER-CARD: the proven cheap engineered signals (expected penalty, bust prob,
    target-row penalty/length, low-card-take flag, 1-ply next penalty, ...).
through a 238 -> 128 -> 64 -> 1 tanh MLP, softmax over the hand; argmax at test
time. Weights come from train_player30.py: BC warm-start on a strong heuristic
(sane init, no cold-start waste) then an on-policy, common-random-number
self-play improvement loop (targets from determinized net rollouts -> no
fixed-teacher ceiling, no distribution shift).

Weights load from `bp30_weights.json`. If absent / shape-mismatched / numpy is
unavailable, the agent falls back to the safe min-expected-penalty heuristic, so
it is never worse than a sane baseline.
"""

import os
import json
import bisect

try:
    import numpy as np
except Exception:
    np = None

_BH = [0]
for _i in range(1, 105):
    if _i == 55:
        _BH.append(7)
    elif _i % 11 == 0:
        _BH.append(5)
    elif _i % 10 == 0:
        _BH.append(3)
    elif _i % 5 == 0:
        _BH.append(2)
    else:
        _BH.append(1)

# input layout: 104 (hand multihot) + 104 (unseen multihot) + 12 (board) +
# 5 (scores) + 2 (progress) = 227 global, + 11 per-card = 238
_N_GLOBAL = 227
_N_PERCARD = 11
_N_FEAT = _N_GLOBAL + _N_PERCARD


def _bust_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Safe fallback policy: min expected immediate penalty (bust-risk aware)."""
    best_i = 0
    best = 1e18
    n_rows = len(ends)
    for i in range(len(hand)):
        c = hand[i]
        tr = -1
        mx = -1
        for r in range(n_rows):
            e = ends[r]
            if mx < e < c:
                mx = e
                tr = r
        if tr == -1:
            score = min_pen + 0.05
        elif lens[tr] >= 5:
            score = pens[tr]
        else:
            lo = bisect.bisect_right(su, mx)
            hi = bisect.bisect_left(su, c)
            gap = hi - lo
            p_cut = gap / nu if nu > 0 else 0.0
            req = 5 - lens[tr]
            if req <= 0 or req > n_other:
                bust = 0.0
            else:
                comb = 1
                for j in range(req):
                    comb = comb * (n_other - j) // (j + 1)
                bust = comb * (p_cut ** req) * ((1.0 - p_cut) ** (n_other - req))
            score = bust * pens[tr] + 0.001 * gap
        if score < best:
            best = score
            best_i = i
    return best_i


def _card_target(c, ends, lens, pens, su, nu, n_other):
    tr = -1
    mx = -1
    for r in range(len(ends)):
        if mx < ends[r] < c:
            mx = ends[r]
            tr = r
    if tr == -1:
        return None
    lo = bisect.bisect_right(su, mx)
    hi = bisect.bisect_left(su, c)
    gap = hi - lo
    if lens[tr] >= 5:
        return (tr, 1.0, gap)
    p_cut = gap / nu if nu > 0 else 0.0
    req = 5 - lens[tr]
    if req <= 0 or req > n_other:
        bust = 0.0
    else:
        comb = 1
        for j in range(req):
            comb = comb * (n_other - j) // (j + 1)
        bust = comb * (p_cut ** req) * ((1.0 - p_cut) ** (n_other - req))
    return (tr, bust, gap)


def _next_min_pen(c, rest, ends, lens, pens):
    e2, l2, p2 = list(ends), list(lens), list(pens)
    tr = -1
    mx = -1
    for r in range(len(e2)):
        if mx < e2[r] < c:
            mx = e2[r]
            tr = r
    if tr == -1:
        vic = min(range(len(p2)), key=lambda r: (p2[r], l2[r], r))
        e2[vic], l2[vic], p2[vic] = c, 1, 0
    elif l2[tr] >= 5:
        e2[tr], l2[tr], p2[tr] = c, 1, 0
    else:
        e2[tr] = c
        l2[tr] += 1
    mp2 = min(p2) if p2 else 0
    best = 1 << 30
    for d in rest:
        ti = -1
        m = -1
        for r in range(len(e2)):
            if m < e2[r] < d:
                m = e2[r]
                ti = r
        if ti == -1:
            pen = mp2
        elif l2[ti] >= 5:
            pen = p2[ti]
        else:
            pen = 0
        if pen < best:
            best = pen
    return best if best != (1 << 30) else 0


def _parse(hand, history, player_idx):
    """Return raw state pieces used by both the heuristic and the encoder."""
    board = history["board"]
    ends = [row[-1] for row in board]
    lens = [len(row) for row in board]
    pens = [sum(_BH[c] for c in row) for row in board]

    scores = history.get("scores", [0, 0, 0, 0])
    n_players = len(scores)
    my_score = scores[player_idx]
    opp_scores = [scores[i] for i in range(n_players) if i != player_idx]
    n_other = len(opp_scores)

    unseen = set(range(1, 105))
    unseen.difference_update(hand)
    for row in board:
        unseen.difference_update(row)
    for past in history.get("history_matrix", []):
        unseen.difference_update(past)
    su = sorted(unseen)
    nu = len(su)
    return (list(hand), ends, lens, pens, su, nu, n_other, my_score,
            opp_scores, unseen, history.get("round", 0))


def _encode(hand, history, player_idx):
    """Build (hand_list, X) where X is an (nC x _N_FEAT) float32 ndarray:
    per-card rows = [shared global block | per-card block]."""
    (hand_list, ends, lens, pens, su, nu, n_other, my_score, opp_scores,
     unseen, rnd) = _parse(hand, history, player_idx)
    nC = len(hand_list)

    g = np.zeros(_N_GLOBAL, dtype=np.float32)
    for c in hand_list:
        g[c - 1] = 1.0                       # 0..103  hand multi-hot
    for c in unseen:
        g[104 + c - 1] = 1.0                 # 104..207 still-live multi-hot
    base = 208
    for r in range(4):
        if r < len(ends):
            g[base + r * 3 + 0] = ends[r] / 104.0
            g[base + r * 3 + 1] = lens[r] / 5.0
            g[base + r * 3 + 2] = pens[r] / 22.0
    base = 220
    mean_opp = (sum(opp_scores) / len(opp_scores)) if opp_scores else 0.0
    g[base + 0] = my_score / 40.0
    g[base + 1] = mean_opp / 40.0
    g[base + 2] = max(-1.5, min(1.5, (my_score - mean_opp) / 20.0))
    g[base + 3] = 1.0 if (opp_scores and my_score < min(opp_scores)) else 0.0
    g[base + 4] = 1.0 if (opp_scores and my_score > max(opp_scores)) else 0.0
    g[225] = rnd / 10.0
    g[226] = nC / 10.0

    mp = min(pens) if pens else 0
    order = sorted(range(nC), key=lambda i: hand_list[i])
    rank_in_hand = [0.0] * nC
    for pos, i in enumerate(order):
        rank_in_hand[i] = pos / (nC - 1) if nC > 1 else 0.0

    X = np.zeros((nC, _N_FEAT), dtype=np.float32)
    X[:, :_N_GLOBAL] = g
    for ci in range(nC):
        c = hand_list[ci]
        info = _card_target(c, ends, lens, pens, su, nu, n_other)
        if info is None:
            exp_pen = mp; bust = 1.0; rowpen = mp; rowlen = 0.0
            gap_end = 1.0; is_take = 1.0; safefit = 0.0
        else:
            tr, bust, _gap = info
            rowpen = pens[tr]
            rowlen = lens[tr] / 5.0
            if lens[tr] >= 5:
                exp_pen = pens[tr]; safefit = 0.0
            else:
                exp_pen = bust * pens[tr]; safefit = 1.0
            gap_end = (c - ends[tr]) / 104.0
            is_take = 0.0
        rest = hand_list[:ci] + hand_list[ci + 1:]
        nxt_pen = _next_min_pen(c, rest, ends, lens, pens) if rest else 0.0
        rows_below = sum(1 for e in ends if e < c) / max(1, len(ends))
        p = _N_GLOBAL
        X[ci, p:p + _N_PERCARD] = (
            exp_pen / 5.0, bust, rowpen / 7.0, rowlen, gap_end, is_take, safefit,
            c / 104.0, rank_in_hand[ci], nxt_pen / 7.0, rows_below,
        )
    return hand_list, X


# ---- weights ----------------------------------------------------------------
_W = None
if np is not None:
    try:
        _wf = os.path.join(os.path.dirname(__file__), "bp30_weights.json")
        if os.path.exists(_wf):
            with open(_wf) as _f:
                _c = json.load(_f)
            if _c.get("W1") and len(_c["W1"][0]) == _N_FEAT:
                _W = {"W1": np.asarray(_c["W1"], dtype=np.float32),
                      "b1": np.asarray(_c["b1"], dtype=np.float32),
                      "W2": np.asarray(_c["W2"], dtype=np.float32),
                      "b2": np.asarray(_c["b2"], dtype=np.float32),
                      "W3": np.asarray(_c["W3"], dtype=np.float32),
                      "b3": np.float32(_c["b3"])}
    except Exception:
        _W = None


def _forward_scores(X, W):
    """X: (nC, D) -> scores (nC,).  238 -> 128 -> 64 -> 1, tanh."""
    H1 = np.tanh(X @ W["W1"].T + W["b1"])
    H2 = np.tanh(H1 @ W["W2"].T + W["b2"])
    return H2 @ W["W3"].reshape(-1) + W["b3"]


class BestPlayer30:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        if _W is None or np is None:
            (hand_list, ends, lens, pens, su, nu, n_other,
             _ms, _os, _un, _rd) = _parse(hand, history, self.player_idx)
            mp = min(pens) if pens else 0
            idx = _bust_pick(hand_list, ends, lens, pens, mp, su, nu, n_other)
            return hand_list[idx]

        hand_list, X = _encode(hand, history, self.player_idx)
        s = _forward_scores(X, _W)
        # tie-break: prefer shedding the higher card
        s = s + 1e-6 * np.asarray(hand_list, dtype=np.float32)
        return hand_list[int(np.argmax(s))]
