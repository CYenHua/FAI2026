import math

class BestPlayer1:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def _calc_bullheads(self, card):
        """計算單張牌的牛頭數量"""
        if card == 55: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _get_auto_take_row(self, board):
        """計算吃掉哪一排最划算"""
        best_idx = 0
        best_pen = float('inf')
        best_len = float('inf')
        
        for i, row in enumerate(board):
            pen = sum(self._calc_bullheads(c) for c in row)
            length = len(row)
            if pen < best_pen or (pen == best_pen and length < best_len):
                best_idx, best_pen, best_len = i, pen, length
                
        return best_idx, best_pen

    def action(self, hand, history):
        board = history["board"]
        row_ends = [row[-1] for row in board]
        
        # --- 1. 建立記憶體：算牌系統 ---
        unseen = set(range(1, 105))
        unseen -= set(hand) # 扣除自己手上的牌
        
        # 扣除目前桌上的牌
        for row in board:
            unseen -= set(row)
            
        # 扣除過去回合大家已經打出的牌 (如果有紀錄的話)
        if "history_matrix" in history:
            for past_round_cards in history["history_matrix"]:
                unseen -= set(past_round_cards)
                
        # 計算對手人數 (總分數陣列長度 - 1)
        opponents = len(history.get("scores", [0]*4)) - 1
        
        best_card = hand[0]
        min_expected_penalty = float('inf')
        cheapest_idx, cheapest_pen = self._get_auto_take_row(board)

        # --- 2. 評估手中每一張牌的風險 ---
        for card in hand:
            expected_penalty = 0
            
            # 情況 A：這張牌比桌上所有牌都小，必定要吃牌
            if card < min(row_ends):
                expected_penalty = cheapest_pen
                # 給予一點額外的小懲罰，鼓勵在安全時先把中介牌出掉
                expected_penalty += 0.1 
            
            # 情況 B：這張牌可以接在某一排後面
            else:
                # 找出會接在哪一排
                target_r_idx = -1
                max_end = -1
                for i, row in enumerate(board):
                    if max_end < row[-1] < card:
                        max_end = row[-1]
                        target_r_idx = i
                        
                target_row = board[target_r_idx]
                current_len = len(target_row)
                row_pen = sum(self._calc_bullheads(c) for c in target_row)
                
                # 計算「空隙 (Gap)」中有幾張未出現的牌
                gap_cards = sum(1 for c in unseen if max_end < c < card)
                
                # 計算對手隨機抽中空隙中卡牌的機率 (簡化模型)
                p_cut = gap_cards / len(unseen) if len(unseen) > 0 else 0
                
                # --- 3. 機率計算：我們剛好成為第 6 張牌的機率 ---
                # 要讓我們成為第 6 張牌，對手必須「剛好」出特定數量的牌插隊在我們前面
                # 我們用二項式分佈來近似： P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
                
                bust_prob = 0
                required_cut_ins = 5 - current_len # 需要幾個人插隊我們才會剛好爆炸
                
                if 0 <= required_cut_ins <= opponents:
                    # 計算恰好有 required_cut_ins 個人插隊的機率
                    bust_prob = math.comb(opponents, required_cut_ins) * (p_cut ** required_cut_ins) * ((1 - p_cut) ** (opponents - required_cut_ins))
                
                # 期望值 = 爆炸機率 * 該排的牛頭數
                expected_penalty = bust_prob * row_pen
                
                # 加上微小的差距懲罰：在預期牛頭數相同時，優先打出最接近排尾的牌 (Gap 越小越好)
                expected_penalty += gap_cards * 0.001

            # 比較並找出風險最低 (期望扣分最少) 的牌
            if expected_penalty < min_expected_penalty:
                min_expected_penalty = expected_penalty
                best_card = card

        return best_card