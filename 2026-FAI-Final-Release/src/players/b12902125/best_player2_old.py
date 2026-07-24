import time
import random

class BestPlayer2:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def _calc_bullheads(self, card):
        """計算單張牌的牛頭數量"""
        if card == 55: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _simulate_trick(self, board, my_card, opp_cards):
        """
        核心模擬引擎：模擬一回合的真實結算過程
        """
        # 建立虛擬桌面 (Deep copy 的輕量快速寫法)
        sim_board = [list(row) for row in board]
        
        # 將我們的牌與對手的隨機牌放進池子裡排序
        played = [(my_card, 0)] # 0 代表我們自己
        for i, c in enumerate(opp_cards):
            played.append((c, i + 1))
            
        played.sort(key=lambda x: x[0])
        my_penalty = 0
        
        # 依序結算每一張牌
        for card, p_idx in played:
            target_r = -1
            max_end = -1
            
            # 尋找最接近的排
            for i, row in enumerate(sim_board):
                if max_end < row[-1] < card:
                    max_end = row[-1]
                    target_r = i
                    
            if target_r == -1:
                # 牌太小，被迫吃掉最便宜的排
                best_idx, best_pen, best_len = 0, float('inf'), float('inf')
                for i, row in enumerate(sim_board):
                    pen = sum(self._calc_bullheads(c) for c in row)
                    # 完全對齊專案規則：牛頭最少 -> 牌數最少 -> 索引最小
                    if pen < best_pen or (pen == best_pen and len(row) < best_len):
                        best_idx, best_pen, best_len = i, pen, len(row)
                
                if p_idx == 0: # 如果是我們吃到，記錄懲罰
                    my_penalty += best_pen
                sim_board[best_idx] = [card]
                
            else:
                # 正常接在排尾
                if len(sim_board[target_r]) == 5:
                    # 第 6 張牌爆盤！
                    if p_idx == 0:
                        my_penalty += sum(self._calc_bullheads(c) for c in sim_board[target_r])
                    sim_board[target_r] = [card]
                else:
                    sim_board[target_r].append(card)
                    
        return my_penalty

    def action(self, hand, history):
        # 只有一張牌時直接出，節省 1 秒鐘的運算資源
        if len(hand) == 1:
            return hand[0]
            
        start_time = time.time()
        board = history["board"]
        
        # 1. 整理未出現的牌 (Determinization 資訊集)
        unseen = set(range(1, 105)) - set(hand)
        for row in board: unseen -= set(row)
        if "history_matrix" in history:
            for past in history["history_matrix"]: unseen -= set(past)
        unseen = list(unseen)
        
        # 紀錄每張手牌的模擬數據
        action_stats = {card: {'visits': 0, 'penalty': 0} for card in hand}
        
        simulations = 0
        # 2. MCTS 模擬迴圈：在 0.85 秒內瘋狂進行模擬，預留 0.15 秒防 Timeout
        while time.time() - start_time < 0.85:
            # 隨機洗牌，發給對手 (決定論 Determinization)
            random.shuffle(unseen)
            opp_cards = unseen[0:3] # 取出 3 張牌假裝是對手這回合要出的牌
            
            # Selection: 平均測試手上的每一張牌
            my_card = hand[simulations % len(hand)]
            
            # Simulation: 透過虛擬遊戲引擎結算這回合
            penalty = self._simulate_trick(board, my_card, opp_cards)
            
            # Backpropagation: 更新節點數據
            action_stats[my_card]['visits'] += 1
            action_stats[my_card]['penalty'] += penalty
            simulations += 1
            
        # 3. 根據模擬結果，選出「平均懲罰期望值最低」的牌
        best_card = hand[0]
        min_avg_penalty = float('inf')
        
        for card in hand:
            if action_stats[card]['visits'] > 0:
                avg_pen = action_stats[card]['penalty'] / action_stats[card]['visits']
                
                # 如果預期懲罰一樣，優先把大的牌出掉 (微小的 Tie-breaker)
                if avg_pen < min_avg_penalty or (avg_pen == min_avg_penalty and card > best_card):
                    min_avg_penalty = avg_pen
                    best_card = card
                    
        return best_card