"""
BestPlayer25 -- ASYMMETRIC self-planning probe. Self-contained.

The plateau: BP13/BP16/BP20 all cap at platform rating ~255 (vs the 268-272 MCTS
leaders), and the opp-strength sweep peaked at BP20's MODERATE (0.5,1) -- deeper
OPPONENT lookahead (BP24, 1.0/4) cratered (175) via budget collapse. So opponent
depth is NOT the missing lever.

BP25 changes a DIFFERENT axis: BP20/22/24 simulate me AND the field with the SAME
policy, but my own continuation is just greedy. Maybe a greedy "me" is the
bottleneck. BP25 keeps the platform-proven sweet-spot OPPONENT model
`_make_strong(0.5, 1)` (cheap, so passes stay high) but lets MY OWN continuation
fully plan its line: `_MY_PICK = _make_strong(1.0, 8)`. Only my ~1/4 of the picks
get costlier, so unlike BP24 the pass count barely drops. Hypothesis: if BP25 >
BP20 the greedy-self was the cap; if BP25 ~= BP20 self-planning isn't the lever
either and the gap is structural (a real my-move tree / better eval).

Knobs: `_MY_PICK` (self-plan depth), `_OPP_PICK` (kept at the sweet spot). Engine
identical to BP20.
"""

import time
import random
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


# --------------------------------------------------------------------------- #
# board mechanics                                                             #
# --------------------------------------------------------------------------- #
def _place(ends, lens, pens, card, n_rows):
    """Apply one card to (ends,lens,pens) IN PLACE; return the bull-head penalty
    the player who laid `card` eats."""
    tr = -1
    mx = -1
    for r in range(n_rows):
        e = ends[r]
        if mx < e < card:
            mx = e
            tr = r
    if tr == -1:
        bi = 0
        bp = pens[0]
        bl = lens[0]
        for r in range(1, n_rows):
            if pens[r] < bp or (pens[r] == bp and lens[r] < bl):
                bi, bp, bl = r, pens[r], lens[r]
        pen = pens[bi]
        ends[bi] = card
        lens[bi] = 1
        pens[bi] = _BH[card]
        return pen
    if lens[tr] >= 5:
        pen = pens[tr]
        ends[tr] = card
        lens[tr] = 1
        pens[tr] = _BH[card]
        return pen
    ends[tr] = card
    lens[tr] += 1
    pens[tr] += _BH[card]
    return 0


def _resolve_all(ends, lens, pens, plays_by_player, n_rows, heads):
    """Resolve a full trick; accumulate each player's eaten heads (0 == me)."""
    for card, who in sorted(plays_by_player):
        heads[who] += _place(ends, lens, pens, card, n_rows)


# --------------------------------------------------------------------------- #
# per-card scoring helpers                                                     #
# --------------------------------------------------------------------------- #
def _bust_score(card, ends, lens, pens, min_pen, su, nu, n_other):
    """Bust-risk-aware expected immediate penalty of laying `card` now.
    0 when it safely extends a row; row pen on a forced/overflow take; otherwise
    P(row gets cut before me) * row pen, plus a tiny gap term to prefer tight
    fits among equals."""
    tr = -1
    mx = -1
    n_rows = len(ends)
    for r in range(n_rows):
        e = ends[r]
        if mx < e < card:
            mx = e
            tr = r
    if tr == -1:
        return min_pen + 0.05
    if lens[tr] >= 5:
        return pens[tr]
    lo = bisect.bisect_right(su, mx)
    hi = bisect.bisect_left(su, card)
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
    return bust * pens[tr] + 0.001 * gap


def _immediate_pen_idx(hand, ends, lens, pens, min_pen, i):
    c = hand[i]
    tr = -1
    mx = -1
    for r in range(len(ends)):
        if mx < ends[r] < c:
            mx = ends[r]
            tr = r
    if tr == -1:
        return min_pen
    if lens[tr] >= 5:
        return pens[tr]
    return 0


def _min_pen_cands(hand, ends, lens, pens, min_pen):
    ps = [_immediate_pen_idx(hand, ends, lens, pens, min_pen, i) for i in range(len(hand))]
    best = min(ps)
    return [i for i in range(len(hand)) if ps[i] == best], best


def _target_end(card, ends):
    best = -1
    for e in ends:
        if best < e < card:
            best = e
    return best


def _greedy_cont_sum(hand, ends, lens, pens, su, nu, n_other, depth):
    """Optimistically play `depth` of my own remaining cards (greedy min bust-aware
    score each, no opponents intervening) and return the SUM of those scores.
    The bust-aware term already prices in opponents cutting rows via the cut
    probability, so this is a cheap proxy for a planning continuation."""
    e, l, p = list(ends), list(lens), list(pens)
    rem = list(hand)
    total = 0.0
    for _ in range(depth):
        if not rem:
            break
        mp = min(p)
        bi = 0
        bs = 1e18
        for k in range(len(rem)):
            s = _bust_score(rem[k], e, l, p, mp, su, nu, n_other)
            if s < bs:
                bs = s
                bi = k
        total += bs
        _place(e, l, p, rem.pop(bi), len(e))
    return total


# --------------------------------------------------------------------------- #
# policies (the swappable lever). Each returns an index into `hand`.           #
# --------------------------------------------------------------------------- #
def _bust_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Min bust-aware immediate penalty (1-ply)."""
    best_i = 0
    best = 1e18
    for i in range(len(hand)):
        s = _bust_score(hand[i], ends, lens, pens, min_pen, su, nu, n_other)
        if s < best:
            best = s
            best_i = i
    return best_i


def _tightfit_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Min immediate penalty; tie-break = smallest gap to a row-end."""
    cand, _ = _min_pen_cands(hand, ends, lens, pens, min_pen)
    if len(cand) == 1:
        return cand[0]

    def key(i):
        e = _target_end(hand[i], ends)
        g = (hand[i] - e) if e >= 0 else 10 ** 6
        return (g, hand[i])
    return min(cand, key=key)


def _lookahead_hi_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Min immediate penalty; tie-break by optimistic next-turn (non-bust) pen,
    then dump the highest card. BP17's strong-baseline proxy; kept as a knob."""
    cand, _ = _min_pen_cands(hand, ends, lens, pens, min_pen)
    if len(cand) == 1:
        return cand[0]

    def key(i):
        e2, l2, p2 = list(ends), list(lens), list(pens)
        _place(e2, l2, p2, hand[i], len(e2))
        mp2 = min(p2) if p2 else 0
        nxt = 1 << 30
        for k in range(len(hand)):
            if k == i:
                continue
            pen = _immediate_pen_idx(hand, e2, l2, p2, mp2, k)
            if pen < nxt:
                nxt = pen
        if nxt == (1 << 30):
            nxt = 0
        e = _target_end(hand[i], ends)
        g = (hand[i] - e) if e >= 0 else 10 ** 6
        return (nxt, g, -hand[i])
    return min(cand, key=key)


def _make_strong(alpha, depth):
    """Factory: a bust-aware policy that minimises immediate bust-score + alpha *
    an optimistic depth-D self-continuation. alpha=0 or depth=0 reduces to
    `_bust_pick`; (0.5, 1) is BP20's default; larger (alpha, depth) = a more
    planning / 'stronger' opponent model."""
    def pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
        n = len(hand)
        best_i = 0
        best = 1e18
        for i in range(n):
            s = _bust_score(hand[i], ends, lens, pens, min_pen, su, nu, n_other)
            if alpha > 0.0 and depth > 0 and n > 1:
                e2, l2, p2 = list(ends), list(lens), list(pens)
                _place(e2, l2, p2, hand[i], len(e2))
                rest = hand[:i] + hand[i + 1:]
                s += alpha * _greedy_cont_sum(rest, e2, l2, p2, su, nu, n_other, depth)
            if s < best:
                best = s
                best_i = i
        return best_i
    return pick


# --------------------------------------------------------------------------- #
# rollout + decision core (parameterised so variants reuse it verbatim)        #
# --------------------------------------------------------------------------- #
def _rollout_all_heads(ends, lens, pens, my_rest, opp_rest, n_rows,
                       su, nu, n_other, heads, my_pick, opp_pick):
    """Play the WHOLE remaining game out: I follow `my_pick`, opponents follow
    `opp_pick`; accumulate every player's eaten heads (0 == me)."""
    my = list(my_rest)
    opps = [list(h) for h in opp_rest]
    while my:
        mp = min(pens)
        mi = my_pick(my, ends, lens, pens, mp, su, nu, n_other)
        plays = [(my.pop(mi), 0)]
        for oi_p, oh in enumerate(opps):
            if oh:
                k = opp_pick(oh, ends, lens, pens, mp, su, nu, n_other)
                plays.append((oh.pop(k), oi_p + 1))
        _resolve_all(ends, lens, pens, plays, n_rows, heads)


def _root_candidates(hand, ends, lens, pens, min_pen, su, nu, n_other, cand_policies):
    """Plausibly-good first cards: min-pen safe plays u lowest u highest u each
    policy's pick. Indices into `hand`."""
    cand, _ = _min_pen_cands(hand, ends, lens, pens, min_pen)
    s = set(cand)
    n = len(hand)
    s.add(min(range(n), key=lambda i: hand[i]))
    s.add(max(range(n), key=lambda i: hand[i]))
    for pol in cand_policies:
        s.add(pol(hand, ends, lens, pens, min_pen, su, nu, n_other))
    return sorted(s)


def decide(player_idx, hand, history, opp_pick, my_pick, cand_policies,
           rank_weight, time_budget, min_passes, max_passes,
           elim, elim_start, elim_every, elim_k):
    """Shared decision engine. Returns the card to play. All knobs are explicit
    params so each variant file passes its own without touching shared globals."""
    if len(hand) == 1:
        return hand[0]

    start = time.time()

    board = history["board"]
    n_rows = len(board)
    ends0 = [row[-1] for row in board]
    lens0 = [len(row) for row in board]
    pens0 = [sum(_BH[c] for c in row) for row in board]
    mp0 = min(pens0)

    scores = history.get("scores", [0, 0, 0, 0])
    n_players = len(scores)
    n_opp = n_players - 1
    n_other = n_opp
    opp_hand_size = len(hand)

    my_start = scores[player_idx]
    opp_start = [scores[i] for i in range(n_players) if i != player_idx]
    start_vec = [my_start] + opp_start

    unseen = set(range(1, 105))
    unseen.difference_update(hand)
    for row in board:
        unseen.difference_update(row)
    for past in history.get("history_matrix", []):
        unseen.difference_update(past)
    unseen = list(unseen)
    su = sorted(unseen)
    nu = len(su)

    hand_list = list(hand)
    cand_idx = _root_candidates(hand_list, ends0, lens0, pens0, mp0,
                                su, nu, n_other, cand_policies)
    m = len(cand_idx)
    if m == 1:
        return hand_list[cand_idx[0]]

    sum_v = [0.0] * m
    sumsq_v = [0.0] * m
    head_sum = [0.0] * m
    rank_sum = [0.0] * m
    cnt = [0] * m
    active = list(range(m))
    passes = 0
    w = rank_weight

    while True:
        if max_passes and passes >= max_passes:
            break
        if passes >= min_passes and time.time() - start > time_budget:
            break

        random.shuffle(unseen)
        opp_first = []
        opp_after = []
        for o in range(n_opp):
            oh = sorted(unseen[o * opp_hand_size:(o + 1) * opp_hand_size])
            oi = opp_pick(oh, ends0, lens0, pens0, mp0, su, nu, n_other)
            opp_first.append(oh[oi])
            opp_after.append(oh[:oi] + oh[oi + 1:])

        for a in active:
            ci = cand_idx[a]
            c = hand_list[ci]
            e = list(ends0); l = list(lens0); p = list(pens0)
            heads = [0] * n_players
            plays = [(c, 0)] + [(opp_first[o], o + 1) for o in range(n_opp)]
            _resolve_all(e, l, p, plays, n_rows, heads)
            rest = hand_list[:ci] + hand_list[ci + 1:]
            _rollout_all_heads(e, l, p, rest, opp_after, n_rows,
                               su, nu, n_other, heads, my_pick, opp_pick)
            my_final = start_vec[0] + heads[0]
            rank = 1.0
            for q in range(1, n_players):
                of = start_vec[q] + heads[q]
                if of < my_final:
                    rank += 1.0
                elif of == my_final:
                    rank += 0.5
            v = (1.0 - w) * heads[0] + w * rank
            sum_v[a] += v
            sumsq_v[a] += v * v
            head_sum[a] += heads[0]
            rank_sum[a] += rank
            cnt[a] += 1
        passes += 1

        if (elim and len(active) > 1 and passes >= elim_start
                and passes % elim_every == 0):
            means = {a: sum_v[a] / cnt[a] for a in active}
            best_a = min(active, key=lambda a: means[a])
            bm = means[best_a]
            kept = []
            for a in active:
                n = cnt[a]
                mean = means[a]
                var = sumsq_v[a] / n - mean * mean
                if var < 0.0:
                    var = 0.0
                se = (var / n) ** 0.5
                if a == best_a or (mean - bm) <= elim_k * se:
                    kept.append(a)
            if kept:
                active = kept

    best_i = cand_idx[0]
    best = 1e18
    for a in range(m):
        if cnt[a] == 0:
            continue
        r = rank_sum[a] / cnt[a]
        h = head_sum[a] / cnt[a]
        v = (1.0 - w) * h + w * r
        v -= 1e-6 * hand_list[cand_idx[a]]   # tie-break: shed the higher card
        if v < best:
            best = v
            best_i = cand_idx[a]
    return hand_list[best_i]


# --------------------------------------------------------------------------- #
# config: asymmetric -- sweet-spot opponent, deep-planning self. See docstring. #
# --------------------------------------------------------------------------- #
_OPP_PICK = _make_strong(0.5, 1)      # platform-proven field model (keep cheap)
_MY_PICK = _make_strong(1.0, 3)       # I plan my own line deeper (the probe)
# NOTE depth=3 chosen to KEEP the pass budget high (~66 vs BP20's ~87 on a 10-card
# move; depth>=4 collapses passes toward BP24's budget-death zone). This isolates
# "does deeper SELF-planning help" from "too few passes". Push depth up later only
# if BP25 already beats BP20 at depth 3.
_RANK_WEIGHT = 0.0
_CAND_POLICIES = (_OPP_PICK, _MY_PICK, _bust_pick)

_TIME_BUDGET = 0.85
_MIN_PASSES = 8
_MAX_PASSES = 0
_ELIM = True
_ELIM_START = 16
_ELIM_EVERY = 8
_ELIM_K = 2.5


class BestPlayer25:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        return decide(self.player_idx, hand, history,
                      opp_pick=_OPP_PICK, my_pick=_MY_PICK,
                      cand_policies=_CAND_POLICIES, rank_weight=_RANK_WEIGHT,
                      time_budget=_TIME_BUDGET, min_passes=_MIN_PASSES,
                      max_passes=_MAX_PASSES, elim=_ELIM, elim_start=_ELIM_START,
                      elim_every=_ELIM_EVERY, elim_k=_ELIM_K)
