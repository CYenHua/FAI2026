import time
import random
# 丟大牌
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
    """Each player picks the card minimizing their own immediate penalty.
    Tiebreak: pen==0 -> dump biggest; forced take -> smallest."""
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


def _place_card_track(board, card, p_idx, row_ends, row_lens, row_pens, min_pen_box, penalties):
    """Place card on board. Updates penalties[p_idx] if that player takes a row."""
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
    """Resolve a trick (4 simultaneous cards, sorted by value)."""
    plays = [(my_card, 0)]
    for i, c in enumerate(opp_cards):
        plays.append((c, i + 1))
    plays.sort()
    for card, p_idx in plays:
        _place_card_track(sim_board, card, p_idx,
                          row_ends, row_lens, row_pens, min_pen_box, penalties)


def _compute_rank(start_pens, end_pens):
    """My rank in [1, n_players]. Ties get midpoint."""
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
                                     base_re, base_rl, base_rp, base_mp, n_opp):
    """First card forced; everyone greedy after. Returns end-of-game penalties array [me, opp..]."""
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
        idx = _greedy_pick_idx(oh, row_ends, row_lens, row_pens, min_pen_box[0])
        opp_cards.append(oh.pop(idx))
    _resolve_trick_track(sim_board, my_first, opp_cards,
                         row_ends, row_lens, row_pens, min_pen_box, penalties)

    while my_h:
        idx = _greedy_pick_idx(my_h, row_ends, row_lens, row_pens, min_pen_box[0])
        mc = my_h.pop(idx)
        opp_cards = []
        for oh in opp_h:
            oi = _greedy_pick_idx(oh, row_ends, row_lens, row_pens, min_pen_box[0])
            opp_cards.append(oh.pop(oi))
        _resolve_trick_track(sim_board, mc, opp_cards,
                             row_ends, row_lens, row_pens, min_pen_box, penalties)

    return penalties


def _search_min_rank(my_h, opp_h_list, sim_board, row_ends, row_lens, row_pens, min_pen,
                     penalties, start_pens):
    """Min rank for me over my-play-orderings, assuming greedy opps."""
    if not my_h:
        return _compute_rank(start_pens, penalties)

    # Opps pick greedily based on current board (they don't see my move)
    opp_cards = []
    new_opp_h = []
    for oh in opp_h_list:
        oi = _greedy_pick_idx(oh, row_ends, row_lens, row_pens, min_pen)
        opp_cards.append(oh[oi])
        new_opp_h.append(oh[:oi] + oh[oi + 1:])

    best_rank = 1e18
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
        leaf_rank = _search_min_rank(my_next, new_opp_h, new_board,
                                     new_re, new_rl, new_rp, new_mp_box[0],
                                     new_pen, start_pens)
        if leaf_rank < best_rank:
            best_rank = leaf_rank

    return best_rank


def _search_per_first_card_rank(hand_list, opp_hands_template, board,
                                base_re, base_rl, base_rp, base_mp,
                                start_pens, n_opp):
    """Returns {first_card: (min_expected_rank, my_total_pen_at_that_min)}."""
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
        new_pen = [0] * (1 + n_opp)

        _resolve_trick_track(new_board, mc, opp_cards,
                             new_re, new_rl, new_rp, new_mp_box, new_pen)
        my_next = hand_list[:i] + hand_list[i + 1:]
        best_rank = _search_min_rank(my_next, new_opp_h, new_board,
                                     new_re, new_rl, new_rp, new_mp_box[0],
                                     new_pen, start_pens)
        result[mc] = best_rank

    return result


_SEARCH_DEPTH_LIMIT = 4


class BestPlayer5:
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

        # Starting cumulative penalties: me first, then opps in their natural order
        me_start = scores[self.player_idx]
        opp_starts = [scores[i] for i in range(n_players) if i != self.player_idx]
        start_pens = [me_start] + opp_starts

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
                per_card_rank = _search_per_first_card_rank(
                    hand_list, opp_hands_template, board,
                    base_row_ends, base_row_lens, base_row_pens, base_min_pen,
                    start_pens, n_opp)
                for c, r in per_card_rank.items():
                    ci = card_to_idx[c]
                    visits[ci] += 1
                    sum_rank[ci] += r
                    # tiebreaker uses my penalty estimate at this min — approximate
                    # by re-deriving via a simple bound: r doesn't carry my_pen, so
                    # we skip the tiebreaker accumulation here (rank suffices).
            else:
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    my_first = hand_list[ci]
                    end_pens = _greedy_rollout_with_first_track(
                        board, hand_list, my_first, opp_hands_template,
                        base_row_ends, base_row_lens, base_row_pens, base_min_pen, n_opp)
                    rank = _compute_rank(start_pens, end_pens)
                    visits[ci] += 1
                    sum_rank[ci] += rank
                    sum_my_pen[ci] += end_pens[0]

        if sum(visits) == 0:
            idx = _greedy_pick_idx(hand_list,
                                   base_row_ends, base_row_lens, base_row_pens, base_min_pen)
            return hand_list[idx]

        # Pick: lowest avg_rank; tiebreak (1) lowest avg my_pen, (2) highest card to dump
        best_i = -1
        best_avg = 1e18
        best_pen = 1e18
        for i in range(n_cands):
            if visits[i] == 0:
                continue
            avg = sum_rank[i] / visits[i]
            pen = sum_my_pen[i] / visits[i] if visits[i] > 0 else 0
            if best_i < 0:
                best_avg = avg
                best_pen = pen
                best_i = i
                continue
            if avg < best_avg - 1e-9:
                best_avg = avg
                best_pen = pen
                best_i = i
            elif abs(avg - best_avg) <= 1e-9:
                if pen < best_pen - 1e-9:
                    best_pen = pen
                    best_i = i
                elif abs(pen - best_pen) <= 1e-9:
                    if hand_list[i] > hand_list[best_i]:
                        best_i = i
        return hand_list[best_i]
