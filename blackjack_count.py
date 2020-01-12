# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 02:42:57 2020

@author: Swifty
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:31:02 2020

@author: Swifty
"""

import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
    
states1 = [(x,0) for x in range(4,22)] + [(x,1) for x in range(12,22)] + [('bust', 0)]
counts = list(range(-3,4))

#states2 = [(x,0,count) for x in range(4,22) for count in counts] + [(x,1,count) for x in range(12,22) for count in counts] + [('bust', 0, 0)]
states3 = [(state, count) for state in states1 for count in counts if not (count != 0  and state == ('bust',0))]

get_ind_reduced = dict([(vals, idx) for idx, vals in enumerate(states1)])
get_state_reduced = dict([(idx, vals) for idx, vals in enumerate(states1)])
get_ind = dict([(vals, idx) for idx, vals in enumerate(states3)])
get_state = dict([(idx, vals) for idx, vals in enumerate(states3)])

def init_rewards(model):
    rewards_matrix = np.zeros((len(states1),len(states1)))
    if model == 1:
        for i in [13,14,15,16,17,23,24,25,26,27]: # Hitting with hand value greater than 17 without ace or hitting on high ace hands
            rewards_matrix[i][rewards_matrix[i] == 0] = -1
        for i in [0,1,2,3,4,5,6,7,8,9,18,19,20]: # Hitting with hand value greater than 17 without ace or hitting on high ace hands
            rewards_matrix[i][rewards_matrix[i] == 0] = 1
        for i in range(28): #reward sticking based on hand value, less than 13 = punishment
            rewards_matrix[i,i] = (get_state_reduced[i][0]-12)
            if i > 17:
                rewards_matrix[i,i] -= 0.5 #sticking slightly worse for aces
        for i in [8,9]: #Neutral
            rewards_matrix[i,i] = 0
        rewards_matrix[:,28] = -5
        
    if model == 2:
        rewards_matrix[:,17] = 2 #Reward states that end up at 21
        rewards_matrix[:,27] = 2
        rewards_matrix[:,28] = -1 #Punish going bust
        
    if model == 3:
        for i in range(28):
            rewards_matrix[i,i] = (get_state_reduced[i][0]**2 - 144)/100
            if i > 17:
                rewards_matrix[i,i] -= 0.5 #sticking slightly worse for aces
        rewards_matrix[:,28] = -1 #Punish going bust
    return rewards_matrix
        
suits=('Hearts','Diamonds','Spades','Clubs')
ranks=('Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King','Ace')
values={'Two':2,'Three':3,'Four':4,'Five':5,'Six':6,'Seven':7,'Eight':8,'Nine':9,'Ten':10,'Jack':10,'Queen':10,'King':10,'Ace':11}
actions = {0:'hit', 1:'stand'}
playing=True

class Card:
       
    def __init__(self,rank,suit):
        self.rank=rank
        self.suit=suit
        
    def __str__(self):
        return f'{self.rank} of {self.suit}'
    
class Deck:
    
    def __init__(self, n):
        self.deck=[]
        
        for suit in suits:
            for rank in ranks:
            
                self.deck.append(Card(rank,suit))
        self.deck=n*self.deck
    
    def __iter__(self):
        return iter(self.deck)
    
    def shuffle(self):
        random.shuffle(self.deck)
        
    def deal(self):
        self.deck.pop()
        return self.deck.pop()
    
class Hand:
    def __init__(self, n):
        self.cards = []
        self.state = ((0, 0), 0)
        self.value = 0   
        self.aces = 0 
        self.low_high = 0
        self.low_high_out = 0
        
    def assign_state(self, deck):
        remaining_decks = np.ceil(len(list(deck))/52)
        if remaining_decks == 0:
            remaining_decks = 1
        sign = np.sign(self.low_high)
        val = abs(self.low_high) / remaining_decks
        if val >= 3:
            self.low_high_out = sign * 3
        elif val >= 2:
            self.low_high_out = sign * 2
        elif val >= 1:
            self.low_high_out = sign * 1
        else:
            self.low_high_out = 0
            
        if self.value > 21:
            self.state = (('bust', 0), 0)
        else:
            self.state = ((self.value, self.aces), self.low_high_out)
    
    def add_card(self,card):
        self.cards.append(card)        
        self.value+=values[card.rank]
        
        if card.rank=='Ace':
            self.aces+=1
            
        if values[card.rank] > 9:
            self.low_high -= 1
            
        elif values[card.rank] < 7:
            self.low_high += 1
        
    def adjust_for_ace(self, deck):
        while self.value > 21 and self.aces:
            self.value -= 10
            self.aces -= 1
            
        self.assign_state(deck)
    
    def new_hand(self, deck):
        self.cards = []
        self.add_card(deck.deal())
        self.add_card(deck.deal())
        self.adjust_for_ace(deck)
            
def hit(deck, hand):
    hand.add_card(deck.deal())
    hand.adjust_for_ace(deck)
    
def hit_or_stand(deck, hand, action):
    while True:
        if action == 'stand':
            return True
        
        elif action == 'hit':
            hit(deck,hand)
            return False
        
        break
    
def take_random_action(random_int, deck, player_hand):
    rand_action = actions[random_int]
    state_index_reduced = get_ind_reduced[player_hand.state[0]]
    stand_interrupt = hit_or_stand(deck, player_hand, rand_action)
    state_index_reduced_new = get_ind_reduced[player_hand.state[0]]
    reward = rewards_matrix[state_index_reduced, state_index_reduced_new]
    state_index_new = get_ind[player_hand.state]
    
    return stand_interrupt, state_index_new, reward

def take_greedy_action(best_action_val, deck, player_hand):
    best_action = actions[best_action_val]
    state_index_reduced = get_ind_reduced[player_hand.state[0]]
    stand_interrupt = hit_or_stand(deck, player_hand, best_action)
    state_index_new = get_ind[player_hand.state]
    state_index_reduced_new = get_ind_reduced[player_hand.state[0]]
    reward = rewards_matrix[state_index_reduced, state_index_reduced_new]
    
    return stand_interrupt, state_index_new, reward

def train(max_iterations, n, learning_rate, greed, discount_factor):
    rand_ints = np.random.randint(0,2,(max_iterations))
    rand_vals = np.random.rand(max_iterations)
    for i in tqdm(range(max_iterations)):
        playing = True
        stand_interrupt = False
        deck = Deck(n)
        deck.shuffle()
        score = 0
        
        while True:
            player_hand = Hand(n)
            player_hand.new_hand(deck)
            
            while playing:
                if len(list(deck)) == 0:
                    # No cards left in deck
                    break
                
                state_index = get_ind[player_hand.state]
                if rand_vals[i] < greed:
                    #Take greedy action
                    best_action_val = np.argmax(Q_table[state_index])
                    stand_interrupt, state_index_new, reward = take_greedy_action(best_action_val, deck, player_hand)
                    Q_curr = Q_table[state_index, best_action_val]
                    action_taken = best_action_val
                    
                else:
                    #Take random action
                    action_int = rand_ints[i]
                    stand_interrupt, state_index_new, reward = take_random_action(action_int, deck, player_hand)
                    Q_curr = Q_table[state_index, action_int]
                    action_taken = action_int
                    
                # Updating Q values - change to use high_low states
                best_next_action = np.argmax(Q_table[state_index_new])
                Q_future = Q_table[state_index_new, best_next_action]
                td_target = reward + discount_factor * Q_future
                td_delta = td_target - Q_curr
                Q_table[state_index, action_taken] = Q_curr + learning_rate * td_delta
                
                playing = not stand_interrupt
                
                if player_hand.value > 21:
                    #Bust
                    break
                
            if player_hand.value <= 21:
                score += player_hand.value ** 2
                
            if len(list(deck)) > 4:
                playing = True
                stand_interrupt = False
            else:
                break
        
def play(Q_table, n, num_plays):
    scores = np.zeros(num_plays)
    for i in range(num_plays):
        playing = True
        stand_interrupt = False
        deck = Deck(n)
        deck.shuffle()
        score = 0
        
        while True:
            player_hand = Hand(n)
            player_hand.new_hand(deck)
            
            while playing:
                if len(list(deck)) == 0: #No cards left in deck
                    break
                
                state_index = get_ind[player_hand.state]
                best_action = actions[np.argmax(Q_table[state_index])]
                stand_interrupt = hit_or_stand(deck, player_hand, best_action)
                
                playing = not stand_interrupt
                
                if player_hand.value > 21: #Bust
                    break
                
            if player_hand.value <= 21:
                score += player_hand.value ** 2
            
            if len(list(deck)) > 4:
                playing = True
                stand_interrupt = False
            else:
                break
        scores[i] = score
    return np.mean(scores)


#%%
#tqdm._instances.clear()
rewards_matrix = init_rewards(3)
max_iterations = 100000
greed = 0.5
discount_factor = 0.5
learning_rate = 0.1         
n = 3   
Q_table = np.zeros((len(states3), 2))
train(max_iterations, n, learning_rate, greed, discount_factor)          
pd.options.display.float_format = '{:,.4f}'.format   
q2 = pd.DataFrame({'states':states3,'hit':Q_table[:,0], 'stand':Q_table[:,1]})
q2
play(Q_table, 2, 10)
q2.to_csv('Q Table with count', sep = ',', float_format = '%.4f')
