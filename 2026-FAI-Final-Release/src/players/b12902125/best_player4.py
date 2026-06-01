import time
import random

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
    """Pick index in `hand` of the card minimizing immediate penalty.
    Tiebreak: on safe plays (pen==0) dump biggest card; on forced takes use smallest."""
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


def _place_card(board, card, is_me, row_ends, row_lens, row_pens, min_pen_box):
    """Mutates board state. Returns penalty for me (0 if not me's row taken)."""
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
        return best_pen if is_me else 0

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
        return pen if is_me else 0

    board[target_r].append(card)
    row_ends[target_r] = card
    row_lens[target_r] += 1
    row_pens[target_r] += _BH[card]
    if row_pens[target_r] < min_pen_box[0]:
        min_pen_box[0] = row_pens[target_r]
    return 0


def _resolve_trick_inplace(sim_board, my_card, opp_cards,
                           row_ends, row_lens, row_pens, min_pen_box):
    plays = [(my_card, True)]
    for c in opp_cards:
        plays.append((c, False))
    plays.sort()
    my_pen = 0
    for card, is_me in plays:
        my_pen += _place_card(sim_board, card, is_me,
                              row_ends, row_lens, row_pens, min_pen_box)
    return my_pen


def _greedy_rollout_with_first(board, hand_list, my_first, opp_hands_template,
                               base_re, base_rl, base_rp, base_mp):
    """Single-card-forced rollout: I play my_first this trick, everyone greedy after."""
    sim_board = [list(r) for r in board]
    row_ends = list(base_re)
    row_lens = list(base_rl)
    row_pens = list(base_rp)
    min_pen_box = [base_mp]

    my_h = [c for c in hand_list if c != my_first]
    opp_h = [list(h) for h in opp_hands_template]

    # First trick: my first card is forced
    opp_cards = []
    for oh in opp_h:
        idx = _greedy_pick_idx(oh, row_ends, row_lens, row_pens, min_pen_box[0])
        opp_cards.append(oh.pop(idx))
    my_pen = _resolve_trick_inplace(sim_board, my_first, opp_cards,
                                    row_ends, row_lens, row_pens, min_pen_box)

    # Remaining tricks: everyone greedy
    while my_h:
        idx = _greedy_pick_idx(my_h, row_ends, row_lens, row_pens, min_pen_box[0])
        mc = my_h.pop(idx)
        opp_cards = []
        for oh in opp_h:
            oi = _greedy_pick_idx(oh, row_ends, row_lens, row_pens, min_pen_box[0])
            opp_cards.append(oh.pop(oi))
        my_pen += _resolve_trick_inplace(sim_board, mc, opp_cards,
                                         row_ends, row_lens, row_pens, min_pen_box)

    return my_pen


def _search_min_pen(my_h, opp_h_list, sim_board,
                    row_ends, row_lens, row_pens, min_pen):
    """Recursive: minimum total penalty if I play optimally and opps play greedily.
    Does NOT mutate inputs (creates copies for each branch)."""
    if not my_h:
        return 0

    # Opps pick greedily based on CURRENT board (simultaneous, so they don't see me)
    opp_cards = []
    new_opp_h = []
    for oh in opp_h_list:
        oi = _greedy_pick_idx(oh, row_ends, row_lens, row_pens, min_pen)
        opp_cards.append(oh[oi])
        new_opp_h.append(oh[:oi] + oh[oi + 1:])

    best = 1 << 30
    for i in range(len(my_h)):
        mc = my_h[i]
        # Fresh state copies for this branch
        new_board = [row[:] for row in sim_board]
        new_re = list(row_ends)
        new_rl = list(row_lens)
        new_rp = list(row_pens)
        new_mp_box = [min_pen]

        pen = _resolve_trick_inplace(new_board, mc, opp_cards,
                                     new_re, new_rl, new_rp, new_mp_box)

        my_next = my_h[:i] + my_h[i + 1:]
        future = _search_min_pen(my_next, new_opp_h, new_board,
                                 new_re, new_rl, new_rp, new_mp_box[0])
        total = pen + future
        if total < best:
            best = total

    return best


def _search_per_first_card(hand_list, opp_hands_template, board,
                           base_re, base_rl, base_rp, base_mp):
    """Returns {first_card: min_total_pen} assuming I play optimally after the first trick."""
    # Opps pick at the current board (they don't see my choice yet)
    opp_cards = []
    new_opp_h = []
    for oh in opp_hands_template:
        oi = _greedy_pick_idx(oh, base_re, base_rl, base_rp, base_mp)
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

        pen = _resolve_trick_inplace(new_board, mc, opp_cards,
                                     new_re, new_rl, new_rp, new_mp_box)
        my_next = hand_list[:i] + hand_list[i + 1:]
        future = _search_min_pen(my_next, new_opp_h, new_board,
                                 new_re, new_rl, new_rp, new_mp_box[0])
        result[mc] = pen + future

    return result


# Endgame search activates when remaining tricks <= this threshold.
# 4 -> 24 leaves per search; 5 -> 120 leaves (too slow per shuffle).
_SEARCH_DEPTH_LIMIT = 4


class BestPlayer4:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        start = time.time()
        TIME_BUDGET = 0.92

        board = history["board"]
        scores = history.get("scores", [0, 0, 0, 0])
        n_players = len(scores)
        n_opp = n_players - 1
        opp_hand_size = len(hand)

        unseen = set(range(1, 105))
        unseen.difference_update(hand)
        for row in board:
            unseen.difference_update(row)
        for past in history.get("history_matrix", []):
            unseen.difference_update(past)
        unseen = list(unseen)

        hand_list = list(hand)
        n_cands = len(hand_list)
        visits = [0] * n_cands
        total_pen = [0.0] * n_cands
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
                # One search returns scores for ALL candidates at once
                per_card_pen = _search_per_first_card(
                    hand_list, opp_hands_template, board,
                    base_row_ends, base_row_lens, base_row_pens, base_min_pen)
                for c, p in per_card_pen.items():
                    ci = card_to_idx[c]
                    visits[ci] += 1
                    total_pen[ci] += p
            else:
                # CRN: evaluate each candidate against the SAME determinization
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    my_first = hand_list[ci]
                    pen = _greedy_rollout_with_first(
                        board, hand_list, my_first, opp_hands_template,
                        base_row_ends, base_row_lens, base_row_pens, base_min_pen)
                    visits[ci] += 1
                    total_pen[ci] += pen

        # Fallback: greedy if no sims
        if sum(visits) == 0:
            idx = _greedy_pick_idx(hand_list,
                                   base_row_ends, base_row_lens, base_row_pens, base_min_pen)
            return hand_list[idx]

        best_i = 0
        best_avg = 1e18
        first = True
        for i in range(n_cands):
            if visits[i] == 0:
                continue
            avg = total_pen[i] / visits[i]
            if first or avg < best_avg or (avg == best_avg and hand_list[i] > hand_list[best_i]):
                best_avg = avg
                best_i = i
                first = False
        return hand_list[best_i]
