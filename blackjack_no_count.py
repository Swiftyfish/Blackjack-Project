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
        
transition_matrix = np.zeros((len(states1),len(states1)))

get_ind = dict([(vals, idx) for idx, vals in enumerate(states1)])
get_state = dict([(idx, vals) for idx, vals in enumerate(states1)])
     
def init_rewards(model):
    rewards_matrix = np.zeros((len(states1),len(states1)))
    if model == 1:
        for i in [13,14,15,16,17,23,24,25,26,27]: # Hitting with hand value greater than 17 without ace or hitting on high ace hands
            rewards_matrix[i][rewards_matrix[i] == 0] = -1
        for i in [0,1,2,3,4,5,6,7,8,9,18,19,20]: # Hitting with hand value greater than 17 without ace or hitting on high ace hands
            rewards_matrix[i][rewards_matrix[i] == 0] = 1
        for i in range(28): #reward sticking based on hand value, less than 13 = punishment
            rewards_matrix[i,i] = (get_state[i][0]-12)
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
            rewards_matrix[i,i] = (get_state[i][0]**2-144)/100
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
    def __init__(self):
        self.cards = []
        self.state = (0,0)
        self.value = 0   
        self.aces = 0 
        self.low_high = 0
        
    def assign_state(self):
        if self.value > 21:
            self.state = ('bust',0)
        else:
            self.state = (self.value, self.aces)
    
    def add_card(self,card):
        self.cards.append(card)        
        self.value+=values[card.rank]
        
        if card.rank=='Ace':
            self.aces+=1
            
        if values[card.rank] > 9:
            self.low_high -= 1
        elif values[card.rank] < 7:
            self.low_high += 1
        
    def adjust_for_ace(self):
        while self.value > 21 and self.aces:
            self.value -= 10
            self.aces -= 1
        self.assign_state()
    
    def new_hand(self, deck):
        self.cards = []
        self.add_card(deck.deal())
        self.add_card(deck.deal())
        self.adjust_for_ace()
            
def hit(deck,hand):
    hand.add_card(deck.deal())
    hand.adjust_for_ace()
    
#def hit_or_stand(deck, hand, action):
#    global stand_interrupt
#
#    while True:
#        if action == 'stand':
#            stand_interrupt = True
#        elif action == 'hit':
#            hit(deck,hand)
#        break

def hit_or_stand(deck, hand, action):
    while True:
        if action == 'stand':
            return True
        elif action == 'hit':
            hit(deck,hand)
            return False
        break
    
def take_random_action(random_int, state_index, deck, player_hand):
    rand_action = actions[random_int]
    stand_interrupt = hit_or_stand(deck, player_hand, rand_action)
    state_index_new = get_ind[player_hand.state]
    reward = rewards_matrix[state_index, state_index_new]
    
    return stand_interrupt, state_index_new, reward

def take_greedy_action(best_action_val, state_index, deck, player_hand):
    best_action = actions[best_action_val]
    stand_interrupt = hit_or_stand(deck, player_hand, best_action)
    state_index_new = get_ind[player_hand.state]
    reward = rewards_matrix[state_index, state_index_new]
    
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
            player_hand = Hand()
            player_hand.new_hand(deck)
            
            while playing:
                if len(list(deck)) == 0:
                    # No cards left in deck
                    break
                
                state_index = get_ind[player_hand.state]
                if rand_vals[i] < greed:
                    #Take greedy action
                    best_action_val = np.argmax(Q_table[state_index])
                    stand_interrupt, state_index_new, reward = take_greedy_action(best_action_val, state_index, deck, player_hand)
                    Q_curr = Q_table[state_index, best_action_val]
                    action_taken = best_action_val
                else:
                    #Take random action
                    action_int = rand_ints[i]
                    stand_interrupt, state_index_new, reward = take_random_action(action_int, state_index, deck, player_hand)
                    Q_curr = Q_table[state_index, action_int]
                    action_taken = action_int
                    
                # Updating Q values
                best_next_action = np.argmax(Q_table[state_index_new])
                Q_future = Q_table[state_index_new, best_next_action]
                # Temporal Difference
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
    hands = 0
    for i in range(num_plays):
        playing = True
        stand_interrupt = False
        deck = Deck(n)
        deck.shuffle()
        score = 0
        
        while True:
            player_hand = Hand()
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
            
            hands += 1
            if player_hand.value <= 21:
                score += player_hand.value ** 2
            
            if len(list(deck)) > 4:
                playing = True
                stand_interrupt = False
            else:
                break
            
        scores[i] = score
    return np.mean(scores), hands


#%%
#tqdm._instances.clear()
rewards_matrix = init_rewards(3)
max_iterations = 100000
greed = 0.9
discount_factor = 0.5
learning_rate = 0.1               
Q_table = np.zeros((len(states1),2))
scores = []
train(max_iterations, 2, learning_rate, greed, discount_factor)  
num_games = 1000 
average_score, num_hands = play(Q_table, 2, num_games)  
average_score/(num_hands/num_games)     
pd.options.display.float_format = '{:,.4f}'.format   
q2 = pd.DataFrame({'states':states1,'hit':Q_table[:,0], 'stand':Q_table[:,1]})
q2
q2.to_csv('Q Table', sep = ',', float_format = '%.4f')
