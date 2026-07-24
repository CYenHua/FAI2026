"""
BestPlayer15 -- a DETERMINISTIC hand-resource scheduler (expert system).

This agent shares *no machinery* with the rollout/search family (BestPlayer8/9/
13) and is also unrelated to the learned linear policy (BestPlayer14). It runs
**no Monte-Carlo simulation, no minimax search, no learning, and no Bayesian
opponent-archetype inference**. It is a closed-form strategic rule system built
on one organising idea borrowed from real 6-Nimmt expert play:

    Treat your ten cards as a *resource to be scheduled over the whole game*,
    not as ten independent one-shot decisions.

Decision principle = EARLIEST-DEADLINE-FIRST on "free" plays:

  * A card is "free" this turn if it extends some row without forcing a take.
  * Each free card has a *deadline*: how soon the rows it could safely land in
    will close (fill to 5, or get undercut by the opponents' unseen cards).
    Cards with few remaining homes and high incoming pressure are the most
    urgent -- play those NOW while they are still free, and keep cards that will
    stay placeable for later. This is scheduling by urgency, not cost-minimising.

  * Low "escape" cards are a reserved resource: when you have one, you don't burn
    it on an ordinary free play, because a low card is your controlled way to
    pick up the *cheapest* row on demand (the engine's low-card rule).

  * When no free play exists (a take is unavoidable this turn) the agent does the
    classic expert reset: take the cheapest row, preferring to do it with a low
    card so the row resets in your favour and your big cards survive.

Because it reasons about the *structure and future* of the hand rather than the
expected value of the current trick, it makes systematically different choices
from the search agents -- ideal as a decorrelated second entry.
"""

import bisect

# Bull-head value table (1..104), identical to the engine.
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


_LOW_ESCAPE = 10   # cards <= this are reserved as controlled-take escapes


class BestPlayer15:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = history["board"]
        n_rows = len(board)
        ends = [row[-1] for row in board]
        lens = [len(row) for row in board]
        pens = [_row_pen(row) for row in board]
        min_pen = min(pens) if pens else 0
        scores = history.get("scores", [0, 0, 0, 0])
        n_opp = len(scores) - 1

        # unseen pool -> used to estimate incoming pressure on each row
        unseen = set(range(1, 105))
        unseen.difference_update(hand)
        for row in board:
            unseen.difference_update(row)
        for past in history.get("history_matrix", []):
            unseen.difference_update(past)
        n_unseen = len(unseen) if unseen else 1
        su = sorted(unseen)
        n_other = n_opp

        # per-row expected opponent cards landing there next round = n_opp *
        # (fraction of unseen cards whose highest end-below is that row).
        row_incoming = [0.0] * n_rows
        for u in unseen:
            tr = -1
            mx = -1
            for r in range(n_rows):
                e = ends[r]
                if mx < e < u:
                    mx = e
                    tr = r
            if tr != -1:
                row_incoming[tr] += 1.0
        pressure = [n_opp * (row_incoming[r] / n_unseen) for r in range(n_rows)]

        hand_list = sorted(hand)
        n_hand = len(hand_list)

        # ---- classify each card: target row, immediate penalty, free? --------
        info = []
        for c in hand_list:
            tr = -1
            mx = -1
            for r in range(n_rows):
                e = ends[r]
                if mx < e < c:
                    mx = e
                    tr = r
            if tr == -1:
                imm = min_pen          # forced low: take cheapest row
                free = False
            elif lens[tr] >= 5:
                imm = pens[tr]         # overflow: take this row now
                free = False
            else:
                imm = 0
                free = True
            info.append((c, tr, imm, free))

        free_cards = [x for x in info if x[3]]

        # ===================== branch A: a free play exists ===================
        # We MINIMISE a cost. Safety (this-trick bust risk) is the primary term
        # so we never spend a card that is about to be busted when a safer free
        # card exists. Among comparably-safe free cards the EARLIEST-DEADLINE
        # scheduling kicks in: play the one whose safe window is closing first,
        # shed higher cards sooner, hold a low escape and a row many of my cards
        # still need. That tie-break layer is the agent's distinctive identity.
        if free_cards:
            best = None
            best_cost = None
            for (c, tr, imm, free) in free_cards:
                slots = 5 - lens[tr]
                homes = sum(1 for r in range(n_rows)
                            if ends[r] < c and lens[r] < 5)
                mine_here = sum(1 for (c2, tr2, _i2, _f2) in info
                                if c2 != c and tr2 == tr)

                # this-trick bust risk = P(row fills to 5 before my card) * row pen
                lo = bisect.bisect_right(su, ends[tr])
                hi = bisect.bisect_left(su, c)
                gap = hi - lo
                p_cut = gap / n_unseen if n_unseen > 0 else 0.0
                req = slots
                if req <= 0 or req > n_other:
                    bust = 0.0
                else:
                    comb = 1
                    for j in range(req):
                        comb = comb * (n_other - j) // (j + 1)
                    bust = comb * (p_cut ** req) * ((1.0 - p_cut) ** (n_other - req))
                bust_risk = bust * pens[tr]

                # deadline: small when the row is closing fast and the card has
                # few alternative homes -> more urgent -> play sooner (lower cost)
                deadline = (slots + 1.0) / (pressure[tr] + 0.5) + 0.7 * (homes - 1)
                urgency = 1.0 / (deadline + 0.5)

                reserve_pen = 0.0
                if c <= _LOW_ESCAPE and len(free_cards) > 1:
                    reserve_pen = 0.15

                cost = (bust_risk                 # PRIMARY: stay safe this trick
                        - 0.22 * urgency          # urgent-first scheduling
                        - 0.013 * c               # shed higher cards sooner
                        + 0.05 * mine_here        # don't hog a row mine need
                        + reserve_pen)            # keep a low escape in reserve
                if best is None or cost < best_cost:
                    best_cost = cost
                    best = c
            return best

        # ================= branch B: a take is unavoidable ====================
        # choose the minimum immediate penalty; among ties do the expert reset:
        # use the lowest card so we reset a cheap row and keep big cards alive.
        min_imm = min(x[2] for x in info)
        cands = [x for x in info if x[2] == min_imm]
        # forced-low cands reset the cheapest row -> prefer the smallest card
        # (true escape); overflow cands -> also prefer smallest to keep highs.
        cands.sort(key=lambda x: x[0])
        return cands[0][0]
