"""
BestPlayer18 -- a genuinely DIFFERENT framework from the BP9/BP13/BP17 family:
regret-matching / decoupled-game (DUCT-style) play.

Every other agent I have best-RESPONDS to an assumed field (BP13 to a per-seat
inferred archetype, BP17 to a single fixed strong policy). This one instead
treats the current trick as a two-player zero-sum MATRIX GAME:

    rows    = which card I play this trick               (I minimise my rank)
    columns = which policy the whole field plays         (field maximises my rank)
    payoff  M[i][j] = my expected final RANK if I play card i and the field
                      follows policy j, estimated by PIMC rollout to game end.

We solve that matrix game with REGRET MATCHING (self-play, average strategy ->
Nash). A single knob `_ADV` slides the field between a fixed realistic prior
(adv=0 -> a regret-matched best response over a field MIXTURE) and the full
strategic adversary (adv=1 -> pure maximin, least exploitable).

MEASURED OUTCOME: more equilibrium = WORSE here. Mean rank rises monotonically
with adv (0.00->2.305, 0.50->2.400, 1.00->2.500), because the real field
(baselines + students) is a fixed, NON-adversarial mixture -- best-responding to
it beats maximin, and full equilibrium wrongly assumes all opponents collude on
the single worst-for-me policy. So the shipped agent runs adv=0: a regret-matched
best response over a 5-policy field MIXTURE. That is still a different framework
and field model than BP13 (per-seat inference) / BP17 (single fixed policy),
which is what makes it a decorrelated best-of-two entry -- but the pure DUCT
equilibrium idea does NOT pay off on this non-adversarial field.

No per-seat Bayesian inference, no minimax tree -- the only search object is the
per-trick matrix game and its regret-matching solution.
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


# -----------------------------------------------------------------------------
# Field / continuation policies (each returns an INDEX into `hand`).
# -----------------------------------------------------------------------------
def _bust_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Min expected immediate penalty, bust-risk aware (the strong baselines)."""
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


def _tightfit_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    cand, _ = _min_pen_cands(hand, ends, lens, pens, min_pen)
    if len(cand) == 1:
        return cand[0]

    def key(i):
        e = _target_end(hand[i], ends)
        g = (hand[i] - e) if e >= 0 else 10 ** 6
        return (g, hand[i])
    return min(cand, key=key)


def _next_min_pen_after(hand, played_idx, ends, lens, pens, min_pen):
    card = hand[played_idx]
    e2, l2, p2 = list(ends), list(lens), list(pens)
    tr = -1
    mx = -1
    for r in range(len(e2)):
        if mx < e2[r] < card:
            mx = e2[r]
            tr = r
    if tr == -1:
        vic = min(range(len(p2)), key=lambda r: (p2[r], l2[r], r))
        e2[vic], l2[vic], p2[vic] = card, 1, 0
    elif l2[tr] >= 5:
        e2[tr], l2[tr], p2[tr] = card, 1, 0
    else:
        e2[tr] = card
        l2[tr] += 1
    mp2 = min(p2) if p2 else 0
    best = 1 << 30
    for k in range(len(hand)):
        if k == played_idx:
            continue
        c = hand[k]
        ti = -1
        m = -1
        for r in range(len(e2)):
            if m < e2[r] < c:
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


def _lookahead_hi_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    cand, _ = _min_pen_cands(hand, ends, lens, pens, min_pen)
    if len(cand) == 1:
        return cand[0]

    def key(i):
        nxt = _next_min_pen_after(hand, i, ends, lens, pens, min_pen)
        e = _target_end(hand[i], ends)
        g = (hand[i] - e) if e >= 0 else 10 ** 6
        return (nxt, g, -hand[i])
    return min(cand, key=key)


def _low_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Always shed the lowest card (a distinct, low-card-hoarding archetype)."""
    bi = 0
    for i in range(1, len(hand)):
        if hand[i] < hand[bi]:
            bi = i
    return bi


def _high_pick(hand, ends, lens, pens, min_pen, su, nu, n_other):
    """Among min-immediate-penalty cards, dump the highest (block archetype)."""
    cand, _ = _min_pen_cands(hand, ends, lens, pens, min_pen)
    bi = cand[0]
    for i in cand:
        if hand[i] > hand[bi]:
            bi = i
    return bi


# Field policy set + a realistic prior over them (weights need not sum to 1; they
# are renormalised). Strong baselines were probed to play bust/lookhi; weaker/mid
# ones tightfit; a minority block (high). low is the contrarian hoarder.
_POLICIES = [_bust_pick, _lookahead_hi_pick, _tightfit_pick, _high_pick, _low_pick]
_PRIOR = [0.30, 0.30, 0.20, 0.12, 0.08]
# My own continuation after the chosen first card (BP17 found lookhi best vs a
# realistic field; it is the protagonist's policy inside every rollout).
_MY_CONT = _lookahead_hi_pick


# -----------------------------------------------------------------------------
# Trick mechanics (engine fit / overflow / low-card rules).
# -----------------------------------------------------------------------------
def _place(ends, lens, pens, card, n_rows):
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
    order = sorted(plays_by_player)
    for card, who in order:
        heads[who] += _place(ends, lens, pens, card, n_rows)


def _rollout_to_end(e, l, p, my_rest, opp_rest, n_rows, su, nu, n_other,
                    field_pick, heads):
    """Play out the rest of the game: me -> _MY_CONT, field -> field_pick.
    Accumulate every player's eaten heads into `heads` IN PLACE (0 == me)."""
    my = list(my_rest)
    opps = [list(h) for h in opp_rest]
    while my:
        mp = min(p)
        mi = _MY_CONT(my, e, l, p, mp, su, nu, n_other)
        plays = [(my.pop(mi), 0)]
        for oi, oh in enumerate(opps):
            if oh:
                k = field_pick(oh, e, l, p, mp, su, nu, n_other)
                plays.append((oh.pop(k), oi + 1))
        _resolve_all(e, l, p, plays, n_rows, heads)


# -----------------------------------------------------------------------------
# Regret-matching solver for the per-trick zero-sum matrix game.
# -----------------------------------------------------------------------------
def _rm_strategy(regret, n):
    pos = [r if r > 0.0 else 0.0 for r in regret]
    s = sum(pos)
    if s > 0.0:
        return [v / s for v in pos]
    return [1.0 / n] * n


def _solve_regret(M, nC, nP, prior, adv, iters):
    """M[i][j] = my expected rank (I minimise, field maximises). Returns my
    average (Nash) strategy over the nC candidate cards. `adv` blends the field
    between the fixed `prior` (0) and its own regret-matching strategy (1)."""
    rc = [0.0] * nC
    rf = [0.0] * nP
    xbar = [0.0] * nC
    for _ in range(iters):
        x = _rm_strategy(rc, nC)
        yf = _rm_strategy(rf, nP)
        y = [(1.0 - adv) * prior[j] + adv * yf[j] for j in range(nP)]
        s = sum(y)
        if s > 0.0:
            y = [v / s for v in y]
        # my action values vs the (blended) field; I want LOW rank
        vi = [sum(y[j] * M[i][j] for j in range(nP)) for i in range(nC)]
        vx = sum(x[i] * vi[i] for i in range(nC))
        for i in range(nC):
            rc[i] += vx - vi[i]          # regret = gain from switching to card i
        # field action values vs my current strategy; field wants HIGH rank
        vj = [sum(x[i] * M[i][j] for i in range(nC)) for j in range(nP)]
        vy = sum(yf[j] * vj[j] for j in range(nP))
        for j in range(nP):
            rf[j] += vj[j] - vy
        for i in range(nC):
            xbar[i] += x[i]
    s = sum(xbar)
    if s > 0.0:
        return [v / s for v in xbar]
    return [1.0 / nC] * nC


# -----------------------------------------------------------------------------
# Knobs (module-level so the CRN eval harness can sweep them).
# -----------------------------------------------------------------------------
# MEASURED (eval_adv18.py, 12 passes, 110 CRN deals vs random baseline triples):
# mean rank rises monotonically with adv -- 0.00->2.305, 0.50->2.400, 1.00->2.500.
# The real field (baselines + students) is a fixed, NON-adversarial mixture, so
# best-responding to it beats maximin; full equilibrium (adv=1) wrongly assumes
# all three opponents collude on the single worst-for-me policy and loses ~0.2.
# So default = 0.0: a regret-matched best response over the 5-policy field
# MIXTURE (still a different framework + field model than BP13/BP17, hence a
# decorrelated best-of-two entry, but NOT naively pessimistic).
_ADV = 0.0            # 0 = robust best-response to prior field, 1 = full maximin
_RM_ITERS = 200       # regret-matching iterations (cheap, matrix is tiny)
_TIME_BUDGET = 0.85   # wall-clock per decision; loop determinizations until up
_MIN_PASSES = 4       # always do at least this many determinizations
_MAX_PASSES = 10 ** 9 # cap (set with _TIME_BUDGET=0 for fixed-pass CRN eval)
_SAMPLE = False       # False = argmax of Nash strategy (low variance); True = sample


def _candidate_indices(hand, ends, lens, pens, su, nu, n_other):
    """The cards any sane strategy might play: all min-immediate-penalty cards,
    plus the extreme low/high cards and each field policy's current pick. Keeps
    the matrix small without dropping any meaningful option."""
    n_rows = len(ends)
    mp = min(pens) if pens else 0
    cand = set(_min_pen_cands(hand, ends, lens, pens, mp)[0])
    cand.add(_low_pick(hand, ends, lens, pens, mp, su, nu, n_other))
    cand.add(_high_pick(hand, ends, lens, pens, mp, su, nu, n_other))
    for pol in _POLICIES:
        cand.add(pol(hand, ends, lens, pens, mp, su, nu, n_other))
    return sorted(cand, key=lambda i: hand[i])


class BestPlayer18:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        start = time.time()

        board = history["board"]
        n_rows = len(board)
        ends0 = [row[-1] for row in board]
        lens0 = [len(row) for row in board]
        pens0 = [sum(_BH[c] for c in row) for row in board]

        scores = history.get("scores", [0, 0, 0, 0])
        n_players = len(scores)
        n_opp = n_players - 1
        n_other = n_opp
        opp_hand_size = len(hand)

        my_start = scores[self.player_idx]
        opp_start = [scores[i] for i in range(n_players) if i != self.player_idx]
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
        cand_idx = _candidate_indices(hand_list, ends0, lens0, pens0, su, nu, n_other)
        nC = len(cand_idx)
        if nC == 1:
            return hand_list[cand_idx[0]]

        nP = len(_POLICIES)
        M_sum = [[0.0] * nP for _ in range(nC)]
        passes = 0

        while True:
            if passes >= _MAX_PASSES:
                break
            if passes >= _MIN_PASSES and (
                    _TIME_BUDGET <= 0 or time.time() - start > _TIME_BUDGET):
                break
            random.shuffle(unseen)
            opp_hands = [sorted(unseen[o * opp_hand_size:(o + 1) * opp_hand_size])
                         for o in range(n_opp)]
            mp0 = min(pens0)

            # opponents' first card this trick under each field policy (the board
            # is the same for every candidate of mine, so compute it once)
            opp_first = [[0] * n_opp for _ in range(nP)]
            opp_after = [[None] * n_opp for _ in range(nP)]
            for j, pol in enumerate(_POLICIES):
                for o, oh in enumerate(opp_hands):
                    k = pol(oh, ends0, lens0, pens0, mp0, su, nu, n_other)
                    opp_first[j][o] = oh[k]
                    opp_after[j][o] = oh[:k] + oh[k + 1:]

            for ci, hi in enumerate(cand_idx):
                c = hand_list[hi]
                rest = hand_list[:hi] + hand_list[hi + 1:]
                for j in range(nP):
                    e = list(ends0); l = list(lens0); p = list(pens0)
                    heads = [0] * n_players
                    plays = [(c, 0)] + [(opp_first[j][o], o + 1) for o in range(n_opp)]
                    _resolve_all(e, l, p, plays, n_rows, heads)
                    _rollout_to_end(e, l, p, rest, opp_after[j], n_rows,
                                    su, nu, n_other, _POLICIES[j], heads)
                    my_final = start_vec[0] + heads[0]
                    rank = 1.0
                    for q in range(1, n_players):
                        of = start_vec[q] + heads[q]
                        if of < my_final:
                            rank += 1.0
                        elif of == my_final:
                            rank += 0.5
                    M_sum[ci][j] += rank
            passes += 1

        inv = 1.0 / passes
        M = [[M_sum[i][j] * inv for j in range(nP)] for i in range(nC)]

        x = _solve_regret(M, nC, nP, _PRIOR, _ADV, _RM_ITERS)

        if _SAMPLE:
            r = random.random()
            acc = 0.0
            best_ci = nC - 1
            for ci in range(nC):
                acc += x[ci]
                if r <= acc:
                    best_ci = ci
                    break
        else:
            best_ci = 0
            best = -1.0
            for ci in range(nC):
                # tie-break toward shedding the higher card (keep low escapes)
                v = x[ci] + 1e-9 * hand_list[cand_idx[ci]]
                if v > best:
                    best = v
                    best_ci = ci
        return hand_list[cand_idx[best_ci]]
