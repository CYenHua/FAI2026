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


def _greedy_pick_idx(hand, board, row_ends, row_lens, row_pens, min_row_pen):
    """Return the index in `hand` of the card minimizing immediate penalty."""
    best_idx = 0
    best_pen = 1 << 30
    best_card = -1
    n_rows = len(board)
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
        # Tiebreak: prefer to dump bigger card when no penalty;
        # when forced to take, prefer the smaller-value card (keep big cards as future blockers).
        if pen < best_pen:
            best_pen = pen
            best_idx = i
            best_card = card
        elif pen == best_pen:
            if pen == 0:
                # safe play — dump the highest card we can
                if card > best_card:
                    best_idx = i
                    best_card = card
            else:
                # we're going to take — pay penalty with smallest card so big cards remain
                if card < best_card:
                    best_idx = i
                    best_card = card
    return best_idx


def _place_card(board, card, is_me, row_ends, row_lens, row_pens, min_row_pen_box):
    """Mutate board, row_ends/lens/pens in place. Returns penalty for `me`."""
    target_r = -1
    max_end = -1
    n_rows = len(board)
    for r in range(n_rows):
        last = row_ends[r]
        if max_end < last < card:
            max_end = last
            target_r = r

    if target_r == -1:
        # Low card: take cheapest row
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
        # recompute min row pen
        m = 1 << 30
        for r in range(n_rows):
            if row_pens[r] < m:
                m = row_pens[r]
        min_row_pen_box[0] = m
        return best_pen if is_me else 0

    if row_lens[target_r] >= 5:
        # 6th card takes the row
        pen = row_pens[target_r]
        board[target_r] = [card]
        row_ends[target_r] = card
        row_lens[target_r] = 1
        row_pens[target_r] = _BH[card]
        m = 1 << 30
        for r in range(n_rows):
            if row_pens[r] < m:
                m = row_pens[r]
        min_row_pen_box[0] = m
        return pen if is_me else 0

    # Normal append
    board[target_r].append(card)
    row_ends[target_r] = card
    row_lens[target_r] += 1
    row_pens[target_r] += _BH[card]
    if row_pens[target_r] < min_row_pen_box[0]:
        min_row_pen_box[0] = row_pens[target_r]
    return 0


def _resolve_trick(sim_board, my_card, opp_cards, row_ends, row_lens, row_pens, min_row_pen_box):
    """Play all 4 cards (sorted by value) and return penalty incurred by me."""
    plays = [(my_card, True)]
    for c in opp_cards:
        plays.append((c, False))
    plays.sort()
    my_pen = 0
    for card, is_me in plays:
        my_pen += _place_card(sim_board, card, is_me,
                              row_ends, row_lens, row_pens, min_row_pen_box)
    return my_pen


class BestPlayer3:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        start = time.time()
        TIME_BUDGET = 0.90

        board = history["board"]
        scores = history.get("scores", [0, 0, 0, 0])
        n_players = len(scores)
        n_opp = n_players - 1
        opp_hand_size = len(hand)  # opponents have same number of cards remaining

        # Unseen pool
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

        # Initial board statistics
        base_row_ends = [row[-1] for row in board]
        base_row_lens = [len(row) for row in board]
        base_row_pens = [_row_pen(row) for row in board]
        base_min_pen = min(base_row_pens) if base_row_pens else 0

        iters = 0
        while True:
            # Time check every iteration (cheap)
            if time.time() - start > TIME_BUDGET:
                break
            iters += 1

            # 1) Determinize: sample opponent hands once for this iteration
            random.shuffle(unseen)
            opp_hands_template = []
            base = 0
            for _o in range(n_opp):
                opp_hands_template.append(unseen[base:base + opp_hand_size])
                base += opp_hand_size

            # 2) Evaluate every candidate against the SAME determinization (CRN)
            for ci in range(n_cands):
                # Time check inside the candidate loop too (prevents timing out
                # mid-iteration when iterations get long late in the game).
                if time.time() - start > TIME_BUDGET:
                    break

                my_first = hand_list[ci]

                # Fresh copies
                sim_board = [list(r) for r in board]
                row_ends = list(base_row_ends)
                row_lens = list(base_row_lens)
                row_pens = list(base_row_pens)
                min_pen_box = [base_min_pen]

                my_h = []
                for c in hand_list:
                    if c != my_first:
                        my_h.append(c)
                opp_h = [list(h) for h in opp_hands_template]

                # ---- First trick: my first card is forced ----
                # Opponents pick greedily (they don't see my card; it's simultaneous)
                opp_cards = []
                for oh in opp_h:
                    idx = _greedy_pick_idx(oh, sim_board,
                                           row_ends, row_lens, row_pens, min_pen_box[0])
                    opp_cards.append(oh.pop(idx))

                my_pen = _resolve_trick(sim_board, my_first, opp_cards,
                                        row_ends, row_lens, row_pens, min_pen_box)

                # ---- Remaining tricks: everyone plays greedily ----
                while my_h:
                    idx = _greedy_pick_idx(my_h, sim_board,
                                           row_ends, row_lens, row_pens, min_pen_box[0])
                    my_card = my_h.pop(idx)
                    opp_cards = []
                    for oh in opp_h:
                        oi = _greedy_pick_idx(oh, sim_board,
                                              row_ends, row_lens, row_pens, min_pen_box[0])
                        opp_cards.append(oh.pop(oi))
                    my_pen += _resolve_trick(sim_board, my_card, opp_cards,
                                             row_ends, row_lens, row_pens, min_pen_box)

                visits[ci] += 1
                total_pen[ci] += my_pen

        # 3) Pick lowest average penalty; on ties, prefer to dump bigger card now
        if sum(visits) == 0:
            # Fallback: greedy single-trick pick
            idx = _greedy_pick_idx(hand_list, board,
                                   base_row_ends, base_row_lens, base_row_pens, base_min_pen)
            return hand_list[idx]

        best_i = -1
        best_avg = 1e18
        for i in range(n_cands):
            if visits[i] == 0:
                continue
            avg = total_pen[i] / visits[i]
            if avg < best_avg or (avg == best_avg and hand_list[i] > hand_list[best_i]):
                best_avg = avg
                best_i = i
        return hand_list[best_i]
