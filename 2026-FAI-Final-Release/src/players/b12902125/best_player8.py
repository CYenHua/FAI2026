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


def _row_pen(row):
    s = 0
    for c in row:
        s += _BH[c]
    return s


def _greedy_pick_idx(hand, row_ends, row_lens, row_pens, min_row_pen):
    """ME-style fallback: dump biggest safe (block opps), take smallest (low tail)."""
    best_idx = 0
    best_pen = 1 << 30
    best_card = -1
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
            pen = min_row_pen
        elif row_lens[target_r] >= 5:
            pen = row_pens[target_r]
        else:
            pen = 0
        if pen < best_pen:
            best_pen = pen
            best_idx = i
            best_card = card
        elif pen == best_pen:
            if pen == 0:
                if card > best_card:
                    best_idx = i
                    best_card = card
            else:
                if card < best_card:
                    best_idx = i
                    best_card = card
    return best_idx


def _smart_pick_idx_opp(hand, row_ends, row_lens, row_pens, min_row_pen,
                        sorted_unseen, n_unseen, n_other_players):
    """Bust-prob aware policy: expected penalty = immediate + binomial bust risk.
    Used for BOTH opponents AND my own rollout continuation so the rollout is
    self-consistent (no asymmetry between a "smart" opponent and a "dumb" me).
    Tiebreak: prefer smaller gap (tight fit)."""
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
            if required < 0 or required > n_other_players:
                bust_prob = 0.0
            else:
                k = required
                n = n_other_players
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


def _greedy_rollout_with_first_track(board, hand_list, my_first, opp_hands_template,
                                     base_re, base_rl, base_rp, base_mp, n_opp,
                                     sorted_unseen, n_unseen, n_other):
    """Self-consistent rollout: ME and OPP both use the bust-prob aware policy."""
    sim_board = [list(r) for r in board]
    row_ends = list(base_re)
    row_lens = list(base_rl)
    row_pens = list(base_rp)
    min_pen_box = [base_mp]

    my_h = [c for c in hand_list if c != my_first]
    opp_h = [list(h) for h in opp_hands_template]
    penalties = [0] * (1 + n_opp)

    opp_cards = []
    for oh in opp_h:
        idx = _smart_pick_idx_opp(oh, row_ends, row_lens, row_pens, min_pen_box[0],
                                  sorted_unseen, n_unseen, n_other)
        opp_cards.append(oh.pop(idx))
    _resolve_trick_track(sim_board, my_first, opp_cards,
                         row_ends, row_lens, row_pens, min_pen_box, penalties)

    while my_h:
        idx = _smart_pick_idx_opp(my_h, row_ends, row_lens, row_pens, min_pen_box[0],
                                  sorted_unseen, n_unseen, n_other)
        mc = my_h.pop(idx)
        opp_cards = []
        for oh in opp_h:
            oi = _smart_pick_idx_opp(oh, row_ends, row_lens, row_pens, min_pen_box[0],
                                     sorted_unseen, n_unseen, n_other)
            opp_cards.append(oh.pop(oi))
        _resolve_trick_track(sim_board, mc, opp_cards,
                             row_ends, row_lens, row_pens, min_pen_box, penalties)

    return penalties


def _search_min_rank(my_h, opp_h_list, sim_board, row_ends, row_lens, row_pens, min_pen,
                     penalties, start_pens, sorted_unseen, n_unseen, n_other):
    """Exhaustive minimax over MY card order (opponents fixed-policy).
    Returns (best_rank, my_penalty_along_that_line). Ties in rank are broken
    internally by lower my-penalty."""
    if not my_h:
        return (_compute_rank(start_pens, penalties), penalties[0])

    opp_cards = []
    new_opp_h = []
    for oh in opp_h_list:
        oi = _smart_pick_idx_opp(oh, row_ends, row_lens, row_pens, min_pen,
                                 sorted_unseen, n_unseen, n_other)
        opp_cards.append(oh[oi])
        new_opp_h.append(oh[:oi] + oh[oi + 1:])

    best_rank = 1e18
    best_pen = 1e18
    for i in range(len(my_h)):
        mc = my_h[i]
        new_board = [row[:] for row in sim_board]
        new_re = list(row_ends)
        new_rl = list(row_lens)
        new_rp = list(row_pens)
        new_mp_box = [min_pen]
        new_pen = list(penalties)

        _resolve_trick_track(new_board, mc, opp_cards,
                             new_re, new_rl, new_rp, new_mp_box, new_pen)

        my_next = my_h[:i] + my_h[i + 1:]
        leaf_rank, leaf_pen = _search_min_rank(my_next, new_opp_h, new_board,
                                               new_re, new_rl, new_rp, new_mp_box[0],
                                               new_pen, start_pens,
                                               sorted_unseen, n_unseen, n_other)
        if leaf_rank < best_rank - 1e-9 or (
                abs(leaf_rank - best_rank) <= 1e-9 and leaf_pen < best_pen):
            best_rank = leaf_rank
            best_pen = leaf_pen

    return (best_rank, best_pen)


def _search_per_first_card_rank(hand_list, opp_hands_template, board,
                                base_re, base_rl, base_rp, base_mp,
                                start_pens, n_opp,
                                sorted_unseen, n_unseen, n_other):
    opp_cards = []
    new_opp_h = []
    for oh in opp_hands_template:
        oi = _smart_pick_idx_opp(oh, base_re, base_rl, base_rp, base_mp,
                                 sorted_unseen, n_unseen, n_other)
        opp_cards.append(oh[oi])
        new_opp_h.append(oh[:oi] + oh[oi + 1:])

    result = {}
    for i in range(len(hand_list)):
        mc = hand_list[i]
        new_board = [row[:] for row in board]
        new_re = list(base_re)
        new_rl = list(base_rl)
        new_rp = list(base_rp)
        new_mp_box = [base_mp]
        new_pen = [0] * (1 + n_opp)

        _resolve_trick_track(new_board, mc, opp_cards,
                             new_re, new_rl, new_rp, new_mp_box, new_pen)
        my_next = hand_list[:i] + hand_list[i + 1:]
        best_rank, best_pen = _search_min_rank(my_next, new_opp_h, new_board,
                                               new_re, new_rl, new_rp, new_mp_box[0],
                                               new_pen, start_pens,
                                               sorted_unseen, n_unseen, n_other)
        result[mc] = (best_rank, best_pen)

    return result


_SEARCH_DEPTH_LIMIT = 4
# Module-level so offline harnesses can lower it for fast fair comparisons;
# unchanged at 0.90 for real play.
_TIME_BUDGET = 0.90
# Smooth tiebreak weight: avg_rank dominates, my-penalty gently breaks near-ties
# to cut the discrete-rank sampling noise. Small enough that a genuine rank gap
# (typically >= 0.05 between distinct cards) is never overridden by penalty.
_PEN_LAMBDA = 0.003


class BestPlayer8:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        start = time.time()
        TIME_BUDGET = _TIME_BUDGET

        board = history["board"]
        scores = history.get("scores", [0, 0, 0, 0])
        n_players = len(scores)
        n_opp = n_players - 1
        n_other = n_players - 1  # opp's view of "other players who might cut"
        opp_hand_size = len(hand)

        me_start = scores[self.player_idx]
        opp_starts = [scores[i] for i in range(n_players) if i != self.player_idx]
        start_pens = [me_start] + opp_starts

        unseen_set = set(range(1, 105))
        unseen_set.difference_update(hand)
        for row in board:
            unseen_set.difference_update(row)
        for past in history.get("history_matrix", []):
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

        base_row_ends = [row[-1] for row in board]
        base_row_lens = [len(row) for row in board]
        base_row_pens = [_row_pen(row) for row in board]
        base_min_pen = min(base_row_pens) if base_row_pens else 0

        use_search = (opp_hand_size <= _SEARCH_DEPTH_LIMIT)

        while True:
            if time.time() - start > TIME_BUDGET:
                break

            random.shuffle(unseen)
            opp_hands_template = []
            base = 0
            for _o in range(n_opp):
                opp_hands_template.append(unseen[base:base + opp_hand_size])
                base += opp_hand_size

            if use_search:
                per_card = _search_per_first_card_rank(
                    hand_list, opp_hands_template, board,
                    base_row_ends, base_row_lens, base_row_pens, base_min_pen,
                    start_pens, n_opp,
                    sorted_unseen, n_unseen, n_other)
                for c, (r, pen) in per_card.items():
                    ci = card_to_idx[c]
                    visits[ci] += 1
                    sum_rank[ci] += r
                    sum_my_pen[ci] += pen
            else:
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    my_first = hand_list[ci]
                    end_pens = _greedy_rollout_with_first_track(
                        board, hand_list, my_first, opp_hands_template,
                        base_row_ends, base_row_lens, base_row_pens, base_min_pen, n_opp,
                        sorted_unseen, n_unseen, n_other)
                    rank = _compute_rank(start_pens, end_pens)
                    visits[ci] += 1
                    sum_rank[ci] += rank
                    sum_my_pen[ci] += end_pens[0]

        if sum(visits) == 0:
            idx = _greedy_pick_idx(hand_list,
                                   base_row_ends, base_row_lens, base_row_pens, base_min_pen)
            return hand_list[idx]

        # Selection: minimize avg_rank + small penalty term (smooth tiebreak).
        best_i = -1
        best_score = 1e18
        best_pen = 1e18
        for i in range(n_cands):
            if visits[i] == 0:
                continue
            avg = sum_rank[i] / visits[i]
            pen = sum_my_pen[i] / visits[i]
            score = avg + _PEN_LAMBDA * pen
            if best_i < 0 or score < best_score - 1e-9:
                best_score = score
                best_pen = pen
                best_i = i
            elif abs(score - best_score) <= 1e-9:
                if pen < best_pen - 1e-9:
                    best_pen = pen
                    best_i = i
                elif abs(pen - best_pen) <= 1e-9:
                    if hand_list[i] > hand_list[best_i]:
                        best_i = i
        return hand_list[best_i]
