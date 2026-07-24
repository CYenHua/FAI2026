import time
import random

# ----------------------------------------------------------------------------
# BestPlayer10 = BestPlayer9's determinized PIMC (per-opponent inference +
# rank-search over sampled "multiverse" worlds) with a RISK-AWARE aggregation
# that directly attacks PIMC's classic strategy-fusion bias.
#
# BP9 picks the card minimizing the *mean* rank across sampled worlds. Because
# the in-world search assumes we (and the placement order) play out perfectly,
# the mean is optimistic: it favours a card that is great on average but
# occasionally collapses us to last place. BP10 keeps every piece of BP9's
# machinery -- the same opponent-belief inference, the same per-opponent
# policy sampling, the same exhaustive low-depth search / rollout -- but scores
# each candidate with
#
#     (1 - beta) * mean_rank  +  beta * CVaR_q(rank)  +  lambda * mean_pen
#
# where CVaR_q is the average rank over the worst (1 - q) fraction of worlds.
# beta=0 reproduces BP9 exactly; beta>0 trades a hair of average for robustness
# against the worlds where strategy fusion would otherwise burn us. All worlds
# are evaluated for every candidate within the SAME determinization (inherited
# from BP9), so the per-card outcome lists are aligned and the tail comparison
# is apples-to-apples.
# ----------------------------------------------------------------------------

try:
    from src.players.b12902125 import best_player9 as bp9
except ImportError:  # pragma: no cover - fallback for direct/relative execution
    import best_player9 as bp9


# ---- tunables (override at module level for sweeps, like bp9._TIME_BUDGET) ---
_TIME_BUDGET = 0.90
_PEN_LAMBDA = bp9._PEN_LAMBDA      # weight on my own accumulated penalty
_RISK_BETA = 0.50                  # 0 -> identical to BP9; >0 -> risk-averse
_RISK_Q = 0.70                     # CVaR tail: average over worst 1-q of worlds
_RISK_ADAPTIVE = True              # scale beta by opponent-belief concentration
_SEARCH_DEPTH_LIMIT = bp9._SEARCH_DEPTH_LIMIT


def _belief_concentration(weights):
    """How sharply we can predict the field, in [0, 1].

    0 = every opponent's posterior is uniform (field looks random / unknown, so
    the in-world search is not trustworthy and risk-aversion just chases noise).
    1 = every opponent is pinned to one policy (search is meaningful, so
    strategy-fusion tail risk is real and worth penalising). Used to gate beta:
    we only pay for CVaR robustness when the search it protects is credible.
    """
    if not weights:
        return 0.0
    npol = bp9._NPOL
    base = 1.0 / npol
    denom = 1.0 - base
    total = 0.0
    for w in weights:
        conc = (max(w) - base) / denom if denom > 0 else 0.0
        if conc < 0.0:
            conc = 0.0
        total += conc
    return total / len(weights)


def _cvar_high(sorted_ranks, q):
    """Mean of the worst (largest) (1 - q) fraction of ranks.

    `sorted_ranks` must be ascending. In 6 Nimmt a higher rank is worse, so the
    risky tail is the top of the sorted list. Always includes at least the
    single worst world so the term is well-defined for few samples.
    """
    n = len(sorted_ranks)
    if n == 0:
        return 0.0
    tail = int(round((1.0 - q) * n))
    if tail < 1:
        tail = 1
    if tail > n:
        tail = n
    s = 0.0
    for k in range(n - tail, n):
        s += sorted_ranks[k]
    return s / tail


class BestPlayer10(bp9.BestPlayer9):
    """PIMC with risk-aware (mean + CVaR) world aggregation.

    Inherits opponent inference (_update_belief / _belief_weights /
    _sample_policy / _reset_belief) and all search/rollout helpers from
    BestPlayer9; only the per-card scoring is changed.
    """

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

        # ---- update opponent belief from observed history (inherited) -------
        board_history = history.get("board_history", [])
        history_matrix = history.get("history_matrix", [])
        try:
            self._update_belief(board_history, history_matrix, hand,
                                n_players, n_rounds)
            weights = self._belief_weights()
        except (ValueError, IndexError):
            weights = None

        # gate the risk term on how predictable the field looks
        if _RISK_ADAPTIVE:
            eff_beta = _RISK_BETA * _belief_concentration(weights)
        else:
            eff_beta = _RISK_BETA

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
        # aligned per-candidate outcome lists (one entry per evaluated world)
        ranks = [[] for _ in range(n_cands)]
        pens = [[] for _ in range(n_cands)]
        card_to_idx = {c: i for i, c in enumerate(hand_list)}

        base_re = [row[-1] for row in board]
        base_rl = [len(row) for row in board]
        base_rp = [bp9._row_pen(row) for row in board]
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

            if weights is None:
                opp_pickers = [bp9._pick_bust] * n_opp
            else:
                opp_pickers = [self._sample_policy(weights[j]) for j in range(n_opp)]

            if use_search:
                per = bp9._search_per_first(
                    hand_list, opp_template, board, base_re, base_rl, base_rp, base_mp,
                    start_pens, n_opp, sorted_unseen, n_unseen, n_other, opp_pickers)
                for c, (r, p) in per.items():
                    ci = card_to_idx[c]
                    ranks[ci].append(r)
                    pens[ci].append(p)
            else:
                for ci in range(n_cands):
                    if time.time() - start > TIME_BUDGET:
                        break
                    pen = bp9._rollout_first(
                        board, hand_list, hand_list[ci], opp_template,
                        base_re, base_rl, base_rp, base_mp, n_opp,
                        sorted_unseen, n_unseen, n_other, opp_pickers)
                    ranks[ci].append(bp9._compute_rank(start_pens, pen))
                    pens[ci].append(pen[0])

        if not any(ranks):
            return hand_list[bp9._pick_bust(hand_list, base_re, base_rl, base_rp,
                                            base_mp, sorted_unseen, n_unseen, n_other)]

        best_i = -1
        best_score = 1e18
        best_pen = 1e18
        for i in range(n_cands):
            v = len(ranks[i])
            if v == 0:
                continue
            mean_rank = sum(ranks[i]) / v
            mean_pen = sum(pens[i]) / v
            if eff_beta > 0.0:
                sr = sorted(ranks[i])
                cvar = _cvar_high(sr, _RISK_Q)
                rank_term = (1.0 - eff_beta) * mean_rank + eff_beta * cvar
            else:
                rank_term = mean_rank
            sc = rank_term + _PEN_LAMBDA * mean_pen
            if best_i < 0 or sc < best_score - 1e-9:
                best_score, best_pen, best_i = sc, mean_pen, i
            elif abs(sc - best_score) <= 1e-9:
                if mean_pen < best_pen - 1e-9:
                    best_pen, best_i = mean_pen, i
                elif abs(mean_pen - best_pen) <= 1e-9 and hand_list[i] > hand_list[best_i]:
                    best_i = i
        return hand_list[best_i]
