import random

class RandomPlayer():
    def __init__(self, player_idx):
        self.player_idx = player_idx
    
    def action(self, hand, history):
        return random.choice(hand)
