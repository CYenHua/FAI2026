import time
import random
import math

# ----------------------------------------------------------------------------
# BestPlayer12 = BestPlayer9 with OPPONENT-HAND-CONSTRAINED determinization.
#
# BP9 samples each opponent's hidden hand by uniformly shuffling the unseen
# cards -- the weakest possible prior. Probing showed every baseline plays
# (near) min-immediate-penalty, so their PAST plays leak information about the
# cards they still hold: if at some past round an opponent played card c, then
# any still-unseen card x that would have had a STRICTLY LOWER immediate penalty
# at that board cannot plausibly be in that opponent's hand (a min-penalty
# player would have played x instead). We turn this into a soft consistency
# weight on each sampled determinization, so the rank estimate is dominated by
# realistic worlds. This is orthogonal to the archetype inference and is a
# strict superset of BP9: with no history every weight is 1.0.
# ----------------------------------------------------------------------------

from src.players.b12902125.best_player9 import (
    BestPlayer9,
    _row_pen,
    _pick_bust,
    _search_per_first,
    _rollout_first,
    _compute_rank,
    _SEARCH_DEPTH_LIMIT,
    _TIME_BUDGET,
    _PEN_LAMBDA,
)

# down-weight factor per inconsistent (opponent, held-card) pair.
# kept mild so lookahead opponents (B6-B10) that occasionally skip a cheaper
# card are only softly penalised, not hard-rejected.
_HAND_K = 0.6
_HAND_TOL = 0  # bull-head penalties are integers; "cheaper" means strictly less


def _imm_pen_card(card, row_ends, row_lens, row_pens, min_row_pen):
    """Immediate penalty of playing `card` onto the given row state."""
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


class BestPlayer12(BestPlayer9):
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

        # ---- per-seat hand feasibility mask from past plays ---------------
        # infeasible[j] = set of unseen cards that opponent seat self._seats[j]
        # could not plausibly still hold under the min-penalty assumption.
        infeasible = None
        if self._seats is not None and history_matrix and board_history:
            seats = self._seats
            masks = [set() for _ in seats]
            t = len(history_matrix)
            for r in range(t):
                if r >= len(board_history):
                    break
                bb = board_history[r]
                re = [row[-1] for row in bb]
                rl = [len(row) for row in bb]
                rp = [_row_pen(row) for row in bb]
                mp = min(rp) if rp else 0
                for j, p in enumerate(seats):
                    c = history_matrix[r][p]
                    pc = _imm_pen_card(c, re, rl, rp, mp)
                    mj = masks[j]
                    for x in sorted_unseen:
                        if x in mj:
                            continue
                        if _imm_pen_card(x, re, rl, rp, mp) < pc - _HAND_TOL:
                            mj.add(x)
            infeasible = masks

        hand_list = list(hand)
        n_cands = len(hand_list)
        visits = [0.0] * n_cands       # now sums of determinization weights
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

            # ---- consistency weight for this determinization --------------
            if infeasible is None:
                w = 1.0
            else:
                viol = 0
                for j in range(n_opp):
                    mj = infeasible[j]
                    if not mj:
                        continue
                    for x in opp_template[j]:
                        if x in mj:
                            viol += 1
                w = math.exp(-_HAND_K * viol) if viol else 1.0

            if use_search:
                per = _search_per_first(
                    hand_list, opp_template, board, base_re, base_rl, base_rp, base_mp,
                    start_pens, n_opp, sorted_unseen, n_unseen, n_other, opp_pickers)
                for c, (r, p) in per.items():
                    ci = card_to_idx[c]
                    visits[ci] += w
                    sum_rank[ci] += w * r
                    sum_my_pen[ci] += w * p
            else:
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    pen = _rollout_first(
                        board, hand_list, hand_list[ci], opp_template,
                        base_re, base_rl, base_rp, base_mp, n_opp,
                        sorted_unseen, n_unseen, n_other, opp_pickers)
                    visits[ci] += w
                    sum_rank[ci] += w * _compute_rank(start_pens, pen)
                    sum_my_pen[ci] += w * pen[0]

        if sum(visits) == 0:
            return hand_list[_pick_bust(hand_list, base_re, base_rl, base_rp, base_mp,
                                        sorted_unseen, n_unseen, n_other)]

        best_i = -1
        best_score = 1e18
        best_pen = 1e18
        for i in range(n_cands):
            if visits[i] <= 0:
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
