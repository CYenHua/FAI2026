"""
BestPlayer28 -- BP27 ISMCTS with fixed-su throughput optimization (su/nu computed once per simulation, not per trick). Single-observer Information-Set MCTS (ISMCTS), the from-scratch
efficient MCTS aiming for the 268-272 classmate tier.

WHY (the evidence-backed pivot)
-------------------------------
* BP13/BP16 (archetype inference + endgame minimax) cap at ~258 platform.
* The flat-PIMC "planner-model" family (BP20..25) is WEAKER (~230): it evaluates
  each first card with a GREEDY self-continuation and has NO search over my own
  move sequence. BP16 beats it precisely because BP16 does exact minimax over the
  last 4 tricks -- i.e. it plans MY line, at least in the endgame.
* Modeling opponents AS planners is a proven dead end (BP26 regressed vs a planner
  field). So the lever is NOT the opponent model -- it is searching MY OWN moves.

WHAT IS NEW
-----------
A genuine UCT tree over MY decision sequence across the WHOLE game (not just the
endgame, not just first-card). Single-observer ISMCTS: nodes are keyed by my own
play-sequence (my information set); each iteration RE-DETERMINIZES the opponents'
hidden hands (uniform over unseen -- robust, per the BP26 lesson that committing
to an opponent model hurts), descends the shared tree by UCB on my card choices,
expands one node, then rolls out to game end with a cheap bust-aware policy for
all seats. Reward = my game rank (heads as tie-break). Statistics are SHARED
across determinizations, so good lines get refined -- this is the mid-game
my-move planning BP20 never had.

Knobs (module-level, patched by evals): _TIME_BUDGET, _MAX_ITERS, _UCT_C,
_RANK_WEIGHT (rank vs heads in the backup), _ROLLOUT (rollout policy factory).
"""

import time
import random
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


# --------------------------------------------------------------------------- #
# board mechanics (compact array form, IN PLACE)                              #
# --------------------------------------------------------------------------- #
def _place(ends, lens, pens, card, n_rows):
    """Apply one card; return the bull-head penalty the layer eats."""
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


def _resolve(ends, lens, pens, plays, n_rows, heads):
    """Resolve a trick: plays = list of (card, who); accumulate eaten heads."""
    plays.sort()
    for card, who in plays:
        heads[who] += _place(ends, lens, pens, card, n_rows)


# --------------------------------------------------------------------------- #
# cheap bust-aware rollout policy (used for opponents and my own rollout)      #
# --------------------------------------------------------------------------- #
def _bust_score(card, ends, lens, pens, min_pen, su, nu, n_other):
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


def _bust_pick(hand, ends, lens, pens, su, nu, n_other):
    mp = min(pens)
    best_i = 0
    best = 1e18
    for i in range(len(hand)):
        s = _bust_score(hand[i], ends, lens, pens, mp, su, nu, n_other)
        if s < best:
            best = s
            best_i = i
    return best_i


# --------------------------------------------------------------------------- #
# ISMCTS                                                                       #
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("N", "children", "child_N", "child_W")

    def __init__(self):
        self.N = 0
        self.children = {}      # card -> _Node
        self.child_N = {}       # card -> visit count
        self.child_W = {}       # card -> total reward


def _rank_reward(my_final, opp_finals, rank_weight):
    """Reward in [0,1]; blends rank (primary) and a small heads term."""
    rank = 1.0
    for of in opp_finals:
        if of < my_final:
            rank += 1.0
        elif of == my_final:
            rank += 0.5
    n = len(opp_finals) + 1
    rank_r = (n - rank) / (n - 1)            # 1 == sole winner, 0 == sole last
    # heads term: fewer eaten heads is better; squashed to [0,1]
    heads_r = 1.0 / (1.0 + 0.06 * my_final)
    return (1.0 - rank_weight) * heads_r + rank_weight * rank_r


def decide(player_idx, hand, history, time_budget, max_iters, uct_c,
           rank_weight, rollout_pick):
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

    my_start = scores[player_idx]
    opp_start = [scores[i] for i in range(n_players) if i != player_idx]

    unseen = set(range(1, 105))
    unseen.difference_update(hand)
    for row in board:
        unseen.difference_update(row)
    for past in history.get("history_matrix", []):
        unseen.difference_update(past)
    unseen = list(unseen)

    hand0 = sorted(hand)
    root = _Node()

    it = 0
    # check the clock every CHECK iterations to amortise time.time()
    CHECK = 32
    while True:
        if max_iters and it >= max_iters:
            break
        if (it & (CHECK - 1)) == 0 and time.time() - start > time_budget:
            break
        it += 1

        # ---- determinize opponents' hidden hands ----
        random.shuffle(unseen)
        opp_hands = []
        b = 0
        for _o in range(n_opp):
            opp_hands.append(unseen[b:b + opp_hand_size])
            b += opp_hand_size

        # mutable game state for this simulation
        ends = list(ends0); lens = list(lens0); pens = list(pens0)
        my_hand = list(hand0)
        # su/nu fixed for the whole simulation (BP20-style) -- avoids an O(nu)
        # re-sort every trick, ~doubling iteration throughput; the cut-probability
        # heuristic only needs the unseen pool, not its per-trick shrinkage.
        su = sorted(unseen)
        nu = len(su)
        heads = [0] * n_players
        # opp seat ids are 1..n_opp in `heads`; my id is 0

        # ---- tree descent + rollout in one downward pass ----
        node = root
        path = []                       # (node, card) chosen inside the tree
        in_tree = True
        while my_hand:
            # choose MY card
            if in_tree:
                untried = [c for c in my_hand if c not in node.child_N]
                if untried:
                    # expand one (bias expansion toward the bust-aware pick)
                    mi = _bust_pick(untried, ends, lens, pens, su, nu, n_other)
                    my_card = untried[mi]
                    child = _Node()
                    node.children[my_card] = child
                    node.child_N[my_card] = 0
                    node.child_W[my_card] = 0.0
                    path.append((node, my_card))
                    node = child
                    in_tree = False     # rollout from here on
                else:
                    # UCB select among tried children
                    logN = math.log(node.N + 1)
                    best_c = None
                    best_u = -1e18
                    for c in my_hand:
                        nC = node.child_N[c]
                        q = node.child_W[c] / nC if nC > 0 else 0.0
                        u = q + uct_c * math.sqrt(logN / (nC + 1))
                        if u > best_u:
                            best_u = u
                            best_c = c
                    my_card = best_c
                    path.append((node, my_card))
                    node = node.children[my_card]
            else:
                my_card = my_hand[rollout_pick(my_hand, ends, lens, pens, su, nu, n_other)]

            # ---- opponents act (policy) and the trick resolves ----
            my_hand.remove(my_card)
            plays = [(my_card, 0)]
            for o in range(n_opp):
                oh = opp_hands[o]
                if oh:
                    oi = rollout_pick(oh, ends, lens, pens, su, nu, n_other)
                    oc = oh[oi]
                    oh.pop(oi)
                    plays.append((oc, o + 1))
            _resolve(ends, lens, pens, plays, n_rows, heads)

        # ---- terminal reward ----
        my_final = my_start + heads[0]
        opp_finals = [opp_start[o] + heads[o + 1] for o in range(n_opp)]
        reward = _rank_reward(my_final, opp_finals, rank_weight)

        # ---- backup along the in-tree path ----
        for nd, c in path:
            nd.N += 1
            nd.child_N[c] += 1
            nd.child_W[c] += reward

    # ---- pick the most robust root child (max visits, Q tie-break) ----
    if not root.child_N:
        return hand0[_bust_pick(hand0, ends0, lens0, pens0,
                                sorted(unseen), len(unseen), n_other)]
    best_card = None
    best_key = None
    for c in root.child_N:
        nC = root.child_N[c]
        q = root.child_W[c] / nC if nC > 0 else 0.0
        key = (nC, q)
        if best_key is None or key > best_key:
            best_key = key
            best_card = c
    return best_card


# --------------------------------------------------------------------------- #
# config / class                                                               #
# --------------------------------------------------------------------------- #
_TIME_BUDGET = 0.85
_MAX_ITERS = 0           # 0 == time-bounded; >0 == fixed iters (timeout-free eval)
_UCT_C = 0.7
_RANK_WEIGHT = 1.0       # optimise rank (platform ranks by avg rank); heads tie-break
_ROLLOUT = _bust_pick


class BestPlayer28:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        return decide(self.player_idx, hand, history,
                      time_budget=_TIME_BUDGET, max_iters=_MAX_ITERS,
                      uct_c=_UCT_C, rank_weight=_RANK_WEIGHT,
                      rollout_pick=_ROLLOUT)
