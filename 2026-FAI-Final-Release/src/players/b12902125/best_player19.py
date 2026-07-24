"""
BestPlayer19 -- a LEARNED-POLICY agent. Completely different test-time machinery
from the BP1-BP18 lineage: there is NO determinized rollout, NO matrix-game
solve, NO opponent inference at decision time. It scores each candidate card
with a tiny trained neural net (1 tanh hidden layer over 16 hand-crafted
features) and plays the argmax in microseconds.

Why this is a genuinely different direction (and a good best-of-two decorrelator):
  * Every other agent SEARCHES at test time (PIMC rollout / regret matching).
    This one is a pure reactive policy -- the "thinking" happened offline during
    training, so its decision surface is shaped by gradient descent, not by a
    hand-written objective + Monte-Carlo estimate. It systematically disagrees
    with the search agents, which is exactly what a best-of-two entry wants.
  * It still LEARNS the proven lever (best-response to the real field): the
    policy is trained by REINFORCE self-play directly against the TA baseline
    triples, optimising final RANK. So it adapts to the actual opponents the way
    BP13's inference does, but via learned weights instead of in-game search.
  * Improves on the old BP14 learning attempt: BP14 coordinate-ascended a linear
    SCORE objective; BP19 does policy-gradient on the true RANK objective over a
    richer feature set with a nonlinear policy.

The trained weights are baked into `_W` below (see train_player19.py). If `_W`
is None the agent falls back to the safe min-immediate-penalty heuristic, so it
is never worse than a sane baseline even untrained.
"""

import math
import bisect

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

_N_FEAT = 18


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
    """Return (target_row, bust_prob, gap_in_unseen) for laying card c, or None
    if c is a low-card take (no smaller row end)."""
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
    """Optimistic next-turn min immediate penalty after I lay c now (the 1-step
    lookahead signal the strong baselines use). Lets the net beat 1-ply play."""
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


def _features(hand_list, ends, lens, pens, su, nu, n_other, my_score, opp_scores):
    """Per-card feature matrix (nC x _N_FEAT). Each row mixes card-specific
    signals with global state so the MLP hidden layer can form interactions."""
    nC = len(hand_list)
    mp = min(pens) if pens else 0
    near_full = sum(1 for L in lens if L >= 4)
    mean_opp = (sum(opp_scores) / len(opp_scores)) if opp_scores else 0.0
    leading = 1.0 if (opp_scores and my_score < min(opp_scores)) else 0.0
    trailing = 1.0 if (opp_scores and my_score > max(opp_scores)) else 0.0
    score_gap = max(-1.5, min(1.5, (my_score - mean_opp) / 20.0))
    frac = nC / 10.0
    order = sorted(range(nC), key=lambda i: hand_list[i])
    rank_in_hand = [0.0] * nC
    for pos, i in enumerate(order):
        rank_in_hand[i] = pos / (nC - 1) if nC > 1 else 0.0

    feats = []
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
        feats.append([
            exp_pen / 5.0, bust, rowpen / 7.0, rowlen, gap_end, is_take, safefit,
            c / 104.0, rank_in_hand[ci],
            mp / 7.0, near_full / 4.0, score_gap, leading, trailing, frac, 1.0,
            nxt_pen / 7.0, rows_below,
        ])
    return feats


# Trained weights: dict with W1 (H x D), b1 (H), W2 (H), b2 (scalar). None => the
# agent uses the safe heuristic fallback. For submission these get baked in below;
# during development they auto-load from the sidecar json train_player19.py writes.
_W = None
_TEMP = 1.0   # softmax temperature (test-time default = argmax, temp unused)

if _W is None:
    try:
        import os as _os
        import json as _json
        _wf = _os.path.join(_os.path.dirname(__file__), "bp19_weights.json")
        if _os.path.exists(_wf):
            with open(_wf) as _f:
                _cand = _json.load(_f)
            # only accept weights whose input width matches the current features
            if _cand["W1"] and len(_cand["W1"][0]) == _N_FEAT:
                _W = _cand
    except Exception:
        _W = None


def _mlp_score(x, W):
    W1 = W["W1"]; b1 = W["b1"]; W2 = W["W2"]; b2 = W["b2"]
    s = b2
    for h in range(len(b1)):
        a = b1[h]
        row = W1[h]
        for d in range(len(x)):
            a += row[d] * x[d]
        s += W2[h] * math.tanh(a)
    return s


def _extract(hand, history, player_idx):
    """Shared by the agent and the trainer: parse the engine state and return
    (hand_list, ends, lens, pens, su, nu, n_other, feats)."""
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

    hand_list = list(hand)
    feats = _features(hand_list, ends, lens, pens, su, nu, n_other,
                      my_score, opp_scores)
    return hand_list, ends, lens, pens, su, nu, n_other, feats


class BestPlayer19:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        hand_list, ends, lens, pens, su, nu, n_other, feats = _extract(
            hand, history, self.player_idx)

        if _W is None:
            mp = min(pens) if pens else 0
            idx = _bust_pick(hand_list, ends, lens, pens, mp, su, nu, n_other)
            return hand_list[idx]

        best_i = 0
        best = -1e18
        for ci in range(len(hand_list)):
            s = _mlp_score(feats[ci], _W)
            # tie-break: among equal scores prefer shedding the higher card
            s += 1e-9 * hand_list[ci]
            if s > best:
                best = s
                best_i = ci
        return hand_list[best_i]
