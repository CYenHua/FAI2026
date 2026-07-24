import time
import random
import bisect
import math

# ----------------------------------------------------------------------------
# BestPlayer13 = BestPlayer9 with a RICHER opponent-archetype library.
#
# BP9 models every opponent's play with one of {bust, block, safelow, lowest}.
# Probing the real baselines (probe_strong.py, FAI-final env) showed the best of
# those four predicts the STRONG baselines poorly: B5 0.64, B8 0.40, B9 0.33,
# i.e. BP9's rollout mis-guesses their cards most of the time -- and getting the
# opponent model right is the dominant lever in this game (worth far more than
# any search tweak; cf. the BP10/BP11/BP12 dead ends, all of which left the
# library unchanged and washed out in noise).
#
# We add two archetypes that the probe showed close the gap, both grounded in
# real baseline behaviour rather than guessed:
#   * tightfit     -- min immediate penalty, tie-break = smallest gap to the
#                     target row-end (B1 0.90->0.97).
#   * lookahead_hi -- min immediate penalty now; among safe ties minimise an
#                     optimistic 1-ply next-turn penalty, then DUMP the highest
#                     card (B5 0.64->0.87, B8 0.40->0.51, B9 0.33->0.41).
#
# Online per-seat inference picks these up automatically when it faces such an
# opponent and ignores them otherwise, so this is a strict superset of BP9 and
# stays overfit-safe for unknown students (the floor keeps a robust mixture).
# Everything else -- determinized rank-search, _MY_PICK rollout, penalty-
# smoothed objective -- is inherited unchanged from BP9.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Inlined from BestPlayer9 (previously imported) so this file is fully
# self-contained -- only one file needs to be submitted. The logic below is an
# exact copy of best_player9.py; nothing has been changed.
# ----------------------------------------------------------------------------

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


def _row_pen(row):
    s = 0
    for c in row:
        s += _BH[c]
    return s


# ============================ archetype policies =============================
# All share the signature (hand, row_ends, row_lens, row_pens, min_row_pen,
# sorted_unseen, n_unseen, n_other) -> index into hand.

def _pick_bust(hand, row_ends, row_lens, row_pens, min_row_pen,
               sorted_unseen, n_unseen, n_other):
    """Bust-prob aware: expected penalty = immediate + binomial bust risk."""
    best_idx = 0
    best_score = 1e18
    n_rows = len(row_ends)
    for i in range(len(hand)):
        card = hand[i]
        target_r = -1
        max_end = -1
        for r in range(n_rows):
            last = row_ends[r]
            if max_end < last < card:
                max_end = last
                target_r = r
        if target_r == -1:
            score = min_row_pen + 0.1
            gap = 0
        elif row_lens[target_r] >= 5:
            score = row_pens[target_r]
            gap = 0
        else:
            row_p = row_pens[target_r]
            cur_len = row_lens[target_r]
            lo_idx = bisect.bisect_right(sorted_unseen, max_end)
            hi_idx = bisect.bisect_left(sorted_unseen, card)
            gap = hi_idx - lo_idx
            p_cut = gap / n_unseen if n_unseen > 0 else 0.0
            required = 5 - cur_len
            if required < 0 or required > n_other:
                bust_prob = 0.0
            else:
                k = required
                n = n_other
                if k == 0:
                    bust_prob = (1.0 - p_cut) ** n
                elif k == n:
                    bust_prob = p_cut ** n
                else:
                    comb = 1
                    for j in range(k):
                        comb = comb * (n - j) // (j + 1)
                    bust_prob = comb * (p_cut ** k) * ((1.0 - p_cut) ** (n - k))
            score = bust_prob * row_p
        score += 0.001 * gap
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


def _immediate_pen(hand, row_ends, row_lens, row_pens, min_row_pen, i):
    card = hand[i]
    target_r = -1
    max_end = -1
    for r in range(len(row_ends)):
        last = row_ends[r]
        if max_end < last < card:
            max_end = last
            target_r = r
    if target_r == -1:
        return min_row_pen
    if row_lens[target_r] >= 5:
        return row_pens[target_r]
    return 0


def _pick_block(hand, row_ends, row_lens, row_pens, min_row_pen,
                sorted_unseen, n_unseen, n_other):
    """Min immediate penalty; on a free/safe play dump the BIGGEST card
    (block opponents, keep low cards), on a forced take play smallest."""
    best_idx = 0
    best_pen = 1 << 30
    best_card = -1
    for i in range(len(hand)):
        pen = _immediate_pen(hand, row_ends, row_lens, row_pens, min_row_pen, i)
        card = hand[i]
        if pen < best_pen:
            best_pen, best_idx, best_card = pen, i, card
        elif pen == best_pen:
            if pen == 0:
                if card > best_card:
                    best_idx, best_card = i, card
            else:
                if card < best_card:
                    best_idx, best_card = i, card
    return best_idx


def _pick_safelow(hand, row_ends, row_lens, row_pens, min_row_pen,
                  sorted_unseen, n_unseen, n_other):
    """Min immediate penalty; always break ties toward the SMALLEST card
    (conserve high cards for more safe options later)."""
    best_idx = 0
    best_pen = 1 << 30
    best_card = 1 << 30
    for i in range(len(hand)):
        pen = _immediate_pen(hand, row_ends, row_lens, row_pens, min_row_pen, i)
        card = hand[i]
        if pen < best_pen or (pen == best_pen and card < best_card):
            best_pen, best_idx, best_card = pen, i, card
    return best_idx


def _pick_lowest(hand, row_ends, row_lens, row_pens, min_row_pen,
                 sorted_unseen, n_unseen, n_other):
    """Naive: always play the lowest card."""
    best_idx = 0
    best_card = hand[0]
    for i in range(1, len(hand)):
        if hand[i] < best_card:
            best_card = hand[i]
            best_idx = i
    return best_idx


_MY_PICK = _pick_bust          # my own rollout continuation (fixed, strong)


# ============================ board placement ================================

def _place_card_track(board, card, p_idx, row_ends, row_lens, row_pens, min_pen_box, penalties):
    target_r = -1
    max_end = -1
    n_rows = len(board)
    for r in range(n_rows):
        last = row_ends[r]
        if max_end < last < card:
            max_end = last
            target_r = r

    if target_r == -1:
        best_i = 0
        best_pen = 1 << 30
        best_len = 1 << 30
        for r in range(n_rows):
            p = row_pens[r]
            ln = row_lens[r]
            if p < best_pen or (p == best_pen and ln < best_len):
                best_i, best_pen, best_len = r, p, ln
        board[best_i] = [card]
        row_ends[best_i] = card
        row_lens[best_i] = 1
        row_pens[best_i] = _BH[card]
        m = 1 << 30
        for r in range(n_rows):
            if row_pens[r] < m:
                m = row_pens[r]
        min_pen_box[0] = m
        penalties[p_idx] += best_pen
        return

    if row_lens[target_r] >= 5:
        pen = row_pens[target_r]
        board[target_r] = [card]
        row_ends[target_r] = card
        row_lens[target_r] = 1
        row_pens[target_r] = _BH[card]
        m = 1 << 30
        for r in range(n_rows):
            if row_pens[r] < m:
                m = row_pens[r]
        min_pen_box[0] = m
        penalties[p_idx] += pen
        return

    board[target_r].append(card)
    row_ends[target_r] = card
    row_lens[target_r] += 1
    row_pens[target_r] += _BH[card]
    if row_pens[target_r] < min_pen_box[0]:
        min_pen_box[0] = row_pens[target_r]


def _resolve_trick_track(sim_board, my_card, opp_cards,
                         row_ends, row_lens, row_pens, min_pen_box, penalties):
    plays = [(my_card, 0)]
    for i, c in enumerate(opp_cards):
        plays.append((c, i + 1))
    plays.sort()
    for card, p_idx in plays:
        _place_card_track(sim_board, card, p_idx,
                          row_ends, row_lens, row_pens, min_pen_box, penalties)


def _compute_rank(start_pens, end_pens):
    my_final = start_pens[0] + end_pens[0]
    rank = 1.0
    n = len(start_pens)
    for k in range(1, n):
        opp_final = start_pens[k] + end_pens[k]
        if opp_final < my_final:
            rank += 1.0
        elif opp_final == my_final:
            rank += 0.5
    return rank


# ============ rollout / search with PER-OPPONENT pickers =====================

def _rollout_first(board, hand_list, my_first, opp_template,
                   base_re, base_rl, base_rp, base_mp, n_opp,
                   sorted_unseen, n_unseen, n_other, opp_pickers):
    sim = [list(r) for r in board]
    re = list(base_re)
    rl = list(base_rl)
    rp = list(base_rp)
    mpb = [base_mp]
    my_h = [c for c in hand_list if c != my_first]
    opp_h = [list(h) for h in opp_template]
    pen = [0] * (1 + n_opp)

    oc = []
    for j, oh in enumerate(opp_h):
        oc.append(oh.pop(opp_pickers[j](oh, re, rl, rp, mpb[0], sorted_unseen, n_unseen, n_other)))
    _resolve_trick_track(sim, my_first, oc, re, rl, rp, mpb, pen)

    while my_h:
        mc = my_h.pop(_MY_PICK(my_h, re, rl, rp, mpb[0], sorted_unseen, n_unseen, n_other))
        oc = []
        for j, oh in enumerate(opp_h):
            oc.append(oh.pop(opp_pickers[j](oh, re, rl, rp, mpb[0], sorted_unseen, n_unseen, n_other)))
        _resolve_trick_track(sim, mc, oc, re, rl, rp, mpb, pen)
    return pen


def _search_min(my_h, opp_h_list, sim, re, rl, rp, mp, pen, start_pens,
                sorted_unseen, n_unseen, n_other, opp_pickers):
    if not my_h:
        return (_compute_rank(start_pens, pen), pen[0])
    oc = []
    new_opp = []
    for j, oh in enumerate(opp_h_list):
        oi = opp_pickers[j](oh, re, rl, rp, mp, sorted_unseen, n_unseen, n_other)
        oc.append(oh[oi])
        new_opp.append(oh[:oi] + oh[oi + 1:])
    best_r = 1e18
    best_p = 1e18
    for i in range(len(my_h)):
        nb = [r[:] for r in sim]
        nre = list(re)
        nrl = list(rl)
        nrp = list(rp)
        nmp = [mp]
        npn = list(pen)
        _resolve_trick_track(nb, my_h[i], oc, nre, nrl, nrp, nmp, npn)
        lr, lp = _search_min(my_h[:i] + my_h[i + 1:], new_opp, nb, nre, nrl, nrp,
                             nmp[0], npn, start_pens,
                             sorted_unseen, n_unseen, n_other, opp_pickers)
        if lr < best_r - 1e-9 or (abs(lr - best_r) <= 1e-9 and lp < best_p):
            best_r, best_p = lr, lp
    return (best_r, best_p)


def _search_per_first(hand_list, opp_template, board, base_re, base_rl, base_rp, base_mp,
                      start_pens, n_opp, sorted_unseen, n_unseen, n_other, opp_pickers):
    oc = []
    new_opp = []
    for j, oh in enumerate(opp_template):
        oi = opp_pickers[j](oh, base_re, base_rl, base_rp, base_mp, sorted_unseen, n_unseen, n_other)
        oc.append(oh[oi])
        new_opp.append(oh[:oi] + oh[oi + 1:])
    res = {}
    for i in range(len(hand_list)):
        nb = [r[:] for r in board]
        nre = list(base_re)
        nrl = list(base_rl)
        nrp = list(base_rp)
        nmp = [base_mp]
        npn = [0] * (1 + n_opp)
        _resolve_trick_track(nb, hand_list[i], oc, nre, nrl, nrp, nmp, npn)
        r, p = _search_min(hand_list[:i] + hand_list[i + 1:], new_opp, nb, nre, nrl, nrp,
                           nmp[0], npn, start_pens,
                           sorted_unseen, n_unseen, n_other, opp_pickers)
        res[hand_list[i]] = (r, p)
    return res


_SEARCH_DEPTH_LIMIT = 4
_PEN_LAMBDA = 0.003


class BestPlayer9:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self._reset_belief()

    def _reset_belief(self):
        self._proc = 0          # rounds already folded into log-likelihood
        self._ll = None         # [opp_slot][policy] accumulated log-likelihood
        self._seats = None      # opponent seats in fixed order

    # ---- in-game opponent inference -------------------------------------
    def _update_belief(self, board_history, history_matrix, my_hand, n_players, n_rounds):
        t = len(history_matrix)
        if self._ll is None or t < self._proc:
            self._reset_belief()
            self._seats = [p for p in range(n_players) if p != self.player_idx]
            self._ll = [[0.0] * _NPOL for _ in self._seats]
        n_other = n_players - 1

        for r in range(self._proc, t):
            if r >= len(board_history):
                break
            board_before = board_history[r]
            re = [row[-1] for row in board_before]
            rl = [len(row) for row in board_before]
            rp = [_row_pen(row) for row in board_before]
            mp = min(rp) if rp else 0

            # cards I could not see at round r -> opponents' possible cards
            seen = set()
            for row in board_before:
                seen.update(row)
            for k in range(r):
                seen.update(history_matrix[k])
            # my own hand at round r = current hand + cards I played in rounds r..t-1
            seen.update(my_hand)
            for k in range(r, t):
                seen.add(history_matrix[k][self.player_idx])
            pool = [x for x in range(1, 105) if x not in seen]
            su = pool  # already sorted ascending
            nu = len(pool)
            rem_len = n_rounds - r
            if rem_len <= 1 or nu == 0:
                continue

            for j, p in enumerate(self._seats):
                c = history_matrix[r][p]
                poolx = [x for x in pool if x != c]
                k_draw = min(rem_len - 1, len(poolx))
                counts = [0] * _NPOL
                for _ in range(_INFER_SAMPLES):
                    others = random.sample(poolx, k_draw) if k_draw > 0 else []
                    h = others + [c]
                    h.sort()
                    for pi in range(_NPOL):
                        idx = _POLICIES[pi](h, re, rl, rp, mp, su, nu, n_other)
                        if h[idx] == c:
                            counts[pi] += 1
                for pi in range(_NPOL):
                    prob = counts[pi] / _INFER_SAMPLES
                    if prob < 0.03:
                        prob = 0.03
                    self._ll[j][pi] += math.log(prob)
        self._proc = t

    def _belief_weights(self):
        """Per opponent slot: normalized posterior over policies, floored."""
        weights = []
        for j in range(len(self._seats)):
            ll = self._ll[j]
            m = max(ll)
            ex = [math.exp(v - m) for v in ll]
            s = sum(ex)
            post = [e / s for e in ex]
            w = [(1 - _BELIEF_FLOOR) * post[pi] + _BELIEF_FLOOR / _NPOL
                 for pi in range(_NPOL)]
            weights.append(w)
        return weights

    @staticmethod
    def _sample_policy(weights):
        x = random.random()
        acc = 0.0
        for pi in range(_NPOL):
            acc += weights[pi]
            if x <= acc:
                return _POLICIES[pi]
        return _POLICIES[-1]

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        start = time.time()
        TIME_BUDGET = _TIME_BUDGET

        board = history["board"]
        scores = history.get("scores", [0, 0, 0, 0])
        n_players = len(scores)
        n_opp = n_players - 1
        n_other = n_players - 1
        opp_hand_size = len(hand)
        n_rounds = opp_hand_size + history.get("round", 0)

        # ---- update opponent belief from observed history -----------------
        board_history = history.get("board_history", [])
        history_matrix = history.get("history_matrix", [])
        try:
            self._update_belief(board_history, history_matrix, hand,
                                 n_players, n_rounds)
            weights = self._belief_weights()
        except (ValueError, IndexError):
            weights = None

        me_start = scores[self.player_idx]
        opp_starts = [scores[i] for i in range(n_players) if i != self.player_idx]
        start_pens = [me_start] + opp_starts

        unseen_set = set(range(1, 105))
        unseen_set.difference_update(hand)
        for row in board:
            unseen_set.difference_update(row)
        for past in history_matrix:
            unseen_set.difference_update(past)
        unseen = list(unseen_set)
        sorted_unseen = sorted(unseen)
        n_unseen = len(unseen)

        hand_list = list(hand)
        n_cands = len(hand_list)
        visits = [0] * n_cands
        sum_rank = [0.0] * n_cands
        sum_my_pen = [0.0] * n_cands
        card_to_idx = {c: i for i, c in enumerate(hand_list)}

        base_re = [row[-1] for row in board]
        base_rl = [len(row) for row in board]
        base_rp = [_row_pen(row) for row in board]
        base_mp = min(base_rp) if base_rp else 0

        use_search = (opp_hand_size <= _SEARCH_DEPTH_LIMIT)

        while True:
            if time.time() - start > TIME_BUDGET:
                break

            random.shuffle(unseen)
            opp_template = []
            base = 0
            for _o in range(n_opp):
                opp_template.append(unseen[base:base + opp_hand_size])
                base += opp_hand_size

            # sample each opponent's policy from its belief (uniform if none)
            if weights is None:
                opp_pickers = [_pick_bust] * n_opp
            else:
                opp_pickers = [self._sample_policy(weights[j]) for j in range(n_opp)]

            if use_search:
                per = _search_per_first(
                    hand_list, opp_template, board, base_re, base_rl, base_rp, base_mp,
                    start_pens, n_opp, sorted_unseen, n_unseen, n_other, opp_pickers)
                for c, (r, p) in per.items():
                    ci = card_to_idx[c]
                    visits[ci] += 1
                    sum_rank[ci] += r
                    sum_my_pen[ci] += p
            else:
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    pen = _rollout_first(
                        board, hand_list, hand_list[ci], opp_template,
                        base_re, base_rl, base_rp, base_mp, n_opp,
                        sorted_unseen, n_unseen, n_other, opp_pickers)
                    visits[ci] += 1
                    sum_rank[ci] += _compute_rank(start_pens, pen)
                    sum_my_pen[ci] += pen[0]

        if sum(visits) == 0:
            return hand_list[_pick_bust(hand_list, base_re, base_rl, base_rp, base_mp,
                                        sorted_unseen, n_unseen, n_other)]

        best_i = -1
        best_score = 1e18
        best_pen = 1e18
        for i in range(n_cands):
            if visits[i] == 0:
                continue
            avg = sum_rank[i] / visits[i]
            pn = sum_my_pen[i] / visits[i]
            sc = avg + _PEN_LAMBDA * pn
            if best_i < 0 or sc < best_score - 1e-9:
                best_score, best_pen, best_i = sc, pn, i
            elif abs(sc - best_score) <= 1e-9:
                if pn < best_pen - 1e-9:
                    best_pen, best_i = pn, i
                elif abs(pn - best_pen) <= 1e-9 and hand_list[i] > hand_list[best_i]:
                    best_i = i
        return hand_list[best_i]

_TIME_BUDGET = 0.90
_BELIEF_FLOOR = 0.06
_INFER_SAMPLES = 6


# ============================ new archetype pickers ==========================

def _target_row_end(card, row_ends):
    """Largest row-end strictly below `card`, or -1 (low-card rule)."""
    best = -1
    for last in row_ends:
        if best < last < card:
            best = last
    return best


def _min_pen_indices(hand, row_ends, row_lens, row_pens, min_row_pen):
    pens = [_immediate_pen(hand, row_ends, row_lens, row_pens, min_row_pen, i)
            for i in range(len(hand))]
    best = min(pens)
    return [i for i in range(len(hand)) if pens[i] == best]


def _next_min_pen_after(hand, played_idx, row_ends, row_lens, row_pens, min_row_pen):
    """Optimistic 1-ply self-lookahead: place hand[played_idx], then return the
    smallest immediate penalty achievable next turn with the remaining hand,
    assuming opponents leave the board untouched."""
    card = hand[played_idx]
    re2, rl2, rp2 = list(row_ends), list(row_lens), list(row_pens)
    tr = _target_row_end_idx(card, re2)
    if tr == -1:
        vic = min(range(len(rp2)), key=lambda r: (rp2[r], rl2[r], r))
        re2[vic], rl2[vic], rp2[vic] = card, 1, 0
    elif rl2[tr] >= 5:
        re2[tr], rl2[tr], rp2[tr] = card, 1, 0
    else:
        re2[tr] = card
        rl2[tr] += 1
    mp2 = min(rp2) if rp2 else 0
    best = 1 << 30
    for k, c in enumerate(hand):
        if k == played_idx:
            continue
        ti = _target_row_end_idx(c, re2)
        if ti == -1:
            pen = mp2
        elif rl2[ti] >= 5:
            pen = rp2[ti]
        else:
            pen = 0
        if pen < best:
            best = pen
    return best if best != (1 << 30) else 0


def _target_row_end_idx(card, row_ends):
    """Index of the row `card` would land on, or -1 for the low-card rule."""
    tr = -1
    me = -1
    for r in range(len(row_ends)):
        if me < row_ends[r] < card:
            me = row_ends[r]
            tr = r
    return tr


def _pick_tightfit(hand, row_ends, row_lens, row_pens, min_row_pen,
                   sorted_unseen, n_unseen, n_other):
    """Min immediate penalty; tie-break toward the smallest gap to a row-end."""
    cand = _min_pen_indices(hand, row_ends, row_lens, row_pens, min_row_pen)
    if len(cand) == 1:
        return cand[0]

    def key(i):
        e = _target_row_end(hand[i], row_ends)
        g = (hand[i] - e) if e >= 0 else 10 ** 6
        return (g, hand[i])
    return min(cand, key=key)


def _pick_lookahead_hi(hand, row_ends, row_lens, row_pens, min_row_pen,
                       sorted_unseen, n_unseen, n_other):
    """Min immediate penalty now; among safe ties minimise optimistic next-turn
    penalty, then dump the HIGHEST card (conserve low cards for flexibility)."""
    cand = _min_pen_indices(hand, row_ends, row_lens, row_pens, min_row_pen)
    if len(cand) == 1:
        return cand[0]

    def key(i):
        nxt = _next_min_pen_after(hand, i, row_ends, row_lens, row_pens, min_row_pen)
        e = _target_row_end(hand[i], row_ends)
        g = (hand[i] - e) if e >= 0 else 10 ** 6
        return (nxt, g, -hand[i])
    return min(cand, key=key)


_POLICIES = [_pick_bust, _pick_block, _pick_safelow, _pick_lowest,
             _pick_tightfit, _pick_lookahead_hi]
_NPOL = len(_POLICIES)

# ---------------------------------------------------------------------------
# BestPlayer16 = BestPlayer13 with a slightly LESS over-committed opponent
# posterior (higher `_BELIEF_FLOOR`).
#
# Honest backstory: I tried to push the 13-line several ways and MEASURED each
# with a common-random-numbers paired eval in its real inference+minimax
# framework (eval_pair16.py / eval_knob16.py). All washed out on held-out deals:
#   * richer/2-ply archetype  -> +0.00 match gain (probe_strong2.py: the library
#                                already saturates what a static archetype can fit;
#                                B6-B10 use planning no archetype captures).
#   * lookahead self-continuation in the deep rollout -> +0.095 ± 0.106 (neutral,
#     if anything worse: BP13 already simulates opponents with realistic inferred
#     archetypes, so a "more realistic me" adds nothing).
#   * _INFER_SAMPLES 6->16 -> -0.005 ± 0.091 (the per-round log-likelihood already
#     self-averages).
#   * _PEN_LAMBDA 0.003->0.006 -> neutral.
# The 13-line is, as the project memory warned, saturated; I will not ship a
# tweak that did not survive held-out testing.
#
# The ONE change kept here is a robustness hedge, not a claimed baseline win:
# `_BELIEF_FLOOR` 0.06 -> 0.10 keeps each opponent's archetype posterior a little
# more mixed (never collapsing hard onto a single MAP archetype). Offline this is
# a WASH vs the TA baselines (pooled ~0.0 over 500 held-out deals across two
# independent seed sets), but a fatter floor is strictly safer against the
# UNKNOWN-student field that actually decides the contest -- over-committing to a
# mis-fit archetype is the failure mode the floor guards. So BP16 is BP13's twin
# with a tuned-for-robustness floor, to A/B on the real classmate platform; if it
# does not help there, BP13 is the fallback and loses nothing.
#
# `_MAX_PASSES` is a pass cap (default unbounded) used only by the offline fixed-
# pass eval so changes can be measured timeout-independently; in the tournament
# the time budget is the only stop. `_MY_PICK` is left at BP13's `_pick_bust`
# (the lookahead variant was tested above and did not help).
# ---------------------------------------------------------------------------
_MAX_PASSES = 10 ** 9
_MY_PICK = _pick_bust
_BELIEF_FLOOR = 0.10


class BestPlayer16(BestPlayer9):
    # ---- belief machinery re-bound to the 6-archetype library --------------
    def _update_belief(self, board_history, history_matrix, my_hand, n_players, n_rounds):
        t = len(history_matrix)
        if self._ll is None or t < self._proc:
            self._reset_belief()
            self._seats = [p for p in range(n_players) if p != self.player_idx]
            self._ll = [[0.0] * _NPOL for _ in self._seats]
        n_other = n_players - 1

        for r in range(self._proc, t):
            if r >= len(board_history):
                break
            board_before = board_history[r]
            re = [row[-1] for row in board_before]
            rl = [len(row) for row in board_before]
            rp = [_row_pen(row) for row in board_before]
            mp = min(rp) if rp else 0

            seen = set()
            for row in board_before:
                seen.update(row)
            for k in range(r):
                seen.update(history_matrix[k])
            seen.update(my_hand)
            for k in range(r, t):
                seen.add(history_matrix[k][self.player_idx])
            pool = [x for x in range(1, 105) if x not in seen]
            su = pool
            nu = len(pool)
            rem_len = n_rounds - r
            if rem_len <= 1 or nu == 0:
                continue

            for j, p in enumerate(self._seats):
                c = history_matrix[r][p]
                poolx = [x for x in pool if x != c]
                k_draw = min(rem_len - 1, len(poolx))
                counts = [0] * _NPOL
                for _ in range(_INFER_SAMPLES):
                    others = random.sample(poolx, k_draw) if k_draw > 0 else []
                    h = others + [c]
                    h.sort()
                    for pi in range(_NPOL):
                        idx = _POLICIES[pi](h, re, rl, rp, mp, su, nu, n_other)
                        if h[idx] == c:
                            counts[pi] += 1
                for pi in range(_NPOL):
                    prob = counts[pi] / _INFER_SAMPLES
                    if prob < 0.03:
                        prob = 0.03
                    self._ll[j][pi] += math.log(prob)
        self._proc = t

    def _belief_weights(self):
        weights = []
        for j in range(len(self._seats)):
            ll = self._ll[j]
            m = max(ll)
            ex = [math.exp(v - m) for v in ll]
            s = sum(ex)
            post = [e / s for e in ex]
            w = [(1 - _BELIEF_FLOOR) * post[pi] + _BELIEF_FLOOR / _NPOL
                 for pi in range(_NPOL)]
            weights.append(w)
        return weights

    @staticmethod
    def _sample_policy(weights):
        x = random.random()
        acc = 0.0
        for pi in range(_NPOL):
            acc += weights[pi]
            if x <= acc:
                return _POLICIES[pi]
        return _POLICIES[-1]

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        start = time.time()
        TIME_BUDGET = _TIME_BUDGET

        board = history["board"]
        scores = history.get("scores", [0, 0, 0, 0])
        n_players = len(scores)
        n_opp = n_players - 1
        n_other = n_players - 1
        opp_hand_size = len(hand)
        n_rounds = opp_hand_size + history.get("round", 0)

        board_history = history.get("board_history", [])
        history_matrix = history.get("history_matrix", [])
        try:
            self._update_belief(board_history, history_matrix, hand,
                                 n_players, n_rounds)
            weights = self._belief_weights()
        except (ValueError, IndexError):
            weights = None

        me_start = scores[self.player_idx]
        opp_starts = [scores[i] for i in range(n_players) if i != self.player_idx]
        start_pens = [me_start] + opp_starts

        unseen_set = set(range(1, 105))
        unseen_set.difference_update(hand)
        for row in board:
            unseen_set.difference_update(row)
        for past in history_matrix:
            unseen_set.difference_update(past)
        unseen = list(unseen_set)
        sorted_unseen = sorted(unseen)
        n_unseen = len(unseen)

        hand_list = list(hand)
        n_cands = len(hand_list)
        visits = [0] * n_cands
        sum_rank = [0.0] * n_cands
        sum_my_pen = [0.0] * n_cands
        card_to_idx = {c: i for i, c in enumerate(hand_list)}

        base_re = [row[-1] for row in board]
        base_rl = [len(row) for row in board]
        base_rp = [_row_pen(row) for row in board]
        base_mp = min(base_rp) if base_rp else 0

        use_search = (opp_hand_size <= _SEARCH_DEPTH_LIMIT)

        passes = 0
        while True:
            if passes >= _MAX_PASSES:
                break
            if time.time() - start > TIME_BUDGET:
                break
            passes += 1

            random.shuffle(unseen)
            opp_template = []
            base = 0
            for _o in range(n_opp):
                opp_template.append(unseen[base:base + opp_hand_size])
                base += opp_hand_size

            if weights is None:
                opp_pickers = [_pick_bust] * n_opp
            else:
                opp_pickers = [self._sample_policy(weights[j]) for j in range(n_opp)]

            if use_search:
                per = _search_per_first(
                    hand_list, opp_template, board, base_re, base_rl, base_rp, base_mp,
                    start_pens, n_opp, sorted_unseen, n_unseen, n_other, opp_pickers)
                for c, (r, p) in per.items():
                    ci = card_to_idx[c]
                    visits[ci] += 1
                    sum_rank[ci] += r
                    sum_my_pen[ci] += p
            else:
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    pen = _rollout_first(
                        board, hand_list, hand_list[ci], opp_template,
                        base_re, base_rl, base_rp, base_mp, n_opp,
                        sorted_unseen, n_unseen, n_other, opp_pickers)
                    visits[ci] += 1
                    sum_rank[ci] += _compute_rank(start_pens, pen)
                    sum_my_pen[ci] += pen[0]

        if sum(visits) == 0:
            return hand_list[_pick_bust(hand_list, base_re, base_rl, base_rp, base_mp,
                                        sorted_unseen, n_unseen, n_other)]

        best_i = -1
        best_score = 1e18
        best_pen = 1e18
        for i in range(n_cands):
            if visits[i] == 0:
                continue
            avg = sum_rank[i] / visits[i]
            pn = sum_my_pen[i] / visits[i]
            sc = avg + _PEN_LAMBDA * pn
            if best_i < 0 or sc < best_score - 1e-9:
                best_score, best_pen, best_i = sc, pn, i
            elif abs(sc - best_score) <= 1e-9:
                if pn < best_pen - 1e-9:
                    best_pen, best_i = pn, i
                elif abs(pn - best_pen) <= 1e-9 and hand_list[i] > hand_list[best_i]:
                    best_i = i
        return hand_list[best_i]
