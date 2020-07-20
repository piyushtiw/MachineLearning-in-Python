#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Piyush Tiwari <ptiwari@ncsu.edu>

import sys
import argparse
import pandas as pd
import numpy as np
import random

random.seed(0)

def not_enough_budget_for_any_advertiser(bid, budgets):
    advertisers = list(bid.keys())
    for advertiser in advertisers:
        if budgets[advertiser] >= bid[advertiser]:
            return False
    return True

class Greedy:
    def __init__(self, budgets, bids, queries):
        self.budgets = budgets
        self.bids = bids
        self.queries = queries

    def run(self):
        revenue = 0.0
        for query in self.queries:
            bidder = self.__find_bidder_greedy(self.bids[query])
            if bidder != -1:
                revenue += self.bids[query][bidder]
                self.budgets[bidder] -= self.bids[query][bidder]

        return revenue;

    def __find_bidder_greedy(self, bids):
        if not_enough_budget_for_any_advertiser(bids, self.budgets):
            return -1;

        keys = list(bids.keys())
        maxBidder = -1
        maxBid = -1;

        for advertiser in keys:
            if self.budgets[advertiser] >= bids[advertiser]:
                if maxBid < bids[advertiser]:
                    maxBidder = advertiser
                    maxBid = bids[advertiser]
                elif maxBid == bids[advertiser]:
                    if maxBidder > advertiser:
                        maxBidder = advertiser
                        maxBid = bids[advertiser]
        return maxBidder


class Balance:
    def __init__(self, budgets, bids, queries):
        self.budgets = budgets
        self.bids = bids
        self.queries = queries

    def run(self):
        revenue = 0.0
        for query in self.queries:
            bidder = self.__find_bidder_balance(self.bids[query])
            if bidder != -1:
                revenue += self.bids[query][bidder]
                self.budgets[bidder] -= self.bids[query][bidder]

        return revenue;

    def __find_bidder_balance(self, bids):
        if not_enough_budget_for_any_advertiser(bids, self.budgets):
            return -1
        
        keys = list(bids.keys())
        maxBidder = -1

        for advertiser in keys:
            if self.budgets[advertiser] >= bids[advertiser]:
                if maxBidder == -1:
                    maxBidder = advertiser
                elif self.budgets[maxBidder] < self.budgets[advertiser]:
                    maxBidder = advertiser
                elif self.budgets[maxBidder] == self.budgets[advertiser]:
                    if maxBidder > advertiser:
                        maxBidder = advertiser

        return maxBidder

class MSVV:
    def __init__(self, budgets, bids, queries):
        self.budgets = budgets
        self.bids = bids
        self.queries = queries
        self.rembudget = budgets

    def run(self):
        revenue = 0.0
        for query in self.queries:
            bidder = self.__find_bidder_msvv(self.bids[query])
            if bidder != -1:
                revenue += self.bids[query][bidder]
                self.rembudget[bidder] -= self.bids[query][bidder]

        return revenue;

    def __find_bidder_msvv(self, bids):
        if not_enough_budget_for_any_advertiser(bids, self.rembudget):
            return -1;

        keys = list(bids.keys())
        maxBidder = keys[0]

        for advertiser in keys:
            if self.rembudget[advertiser] >= bids[advertiser]:
                m1 = self.__scaledBid(bids[maxBidder], self.rembudget[maxBidder], self.budgets[maxBidder])
                m2 = self.__scaledBid(bids[advertiser], self.rembudget[advertiser], self.budgets[advertiser])
                if m1 < m2:
                    maxBidder = advertiser
                elif m1 == m2:
                    if maxBidder > advertiser:
                        maxBidder = advertiser

        return maxBidder

    def __scaledBid (self, bid, rembud, bud):
        def psi (xu):
            return 1 - np.exp(xu-1)
        
        xu = (bud-rembud)/bud
        return bid*psi(xu)

#Function to parse input paramteres
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str, help='Algorithm to apply')
    
    return parser.parse_args()

#class to run the algorithm based on input paramteres
class Algorithm:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.budget, self.bids, self.queries = self.__load_data()
        
    def calculate_revenue(self):
        total_revenue = 0.0;
        iters = 100;

        for i in range(iters):
            random.shuffle(self.queries)
            if self.algorithm == "greedy":
                revenue = self.__greedy()
            if self.algorithm == "balance":
                revenue = self.__balance()
            if self.algorithm == "msvv":
                revenue = self.__msvv()

            total_revenue += revenue

        r = total_revenue/iters
        print(r)
        print (r/sum(self.budget.values()))

    def __load_data(self):
        bidder_dataset = pd.read_csv('bidder_dataset.csv')

        budget = {}
        bids = {}

        for row in range(len(bidder_dataset)):
            advertiser = bidder_dataset.iloc[row]['Advertiser']
            keyword = bidder_dataset.iloc[row]['Keyword']
            bid_value = bidder_dataset.iloc[row]['Bid Value']
            budget_value = bidder_dataset.iloc[row]['Budget']
            
            if not (advertiser in budget):
                budget[advertiser] = budget_value
            if not (keyword in bids):
                bids[keyword] = {}
            if not (advertiser in bids[keyword]):
                bids[keyword][advertiser] = bid_value

        with open('queries.txt') as file:
            queries = file.readlines()
        queries = [query.strip() for query in queries]

        return budget, bids, queries


    def __greedy(self):
        algo = Greedy(dict(self.budget), self.bids, self.queries)
        return algo.run()

    def __balance(self):
        algo = Balance(dict(self.budget), self.bids, self.queries)
        return algo.run()

    def __msvv(self):
        algo = MSVV(dict(self.budget), self.bids, self.queries)
        return algo.run()

if __name__ == '__main__':
    Args = get_args()
    
    algo = Algorithm(Args.algorithm)
    algo.calculate_revenue()
