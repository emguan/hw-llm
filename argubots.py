"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files

###########################################
# Define your own additional argubots here!
###########################################

from rank_bm25 import BM25Okapi as BM25_Index 
import numpy as np

class AkikiBot(KialoAgent):

    def __init__(self, name: str, kialo: Kialo, kind='all'):
        super().__init__(name, kialo)
        self.kind = kind

    def response(self, d: Dialogue) -> str:

        # First turn
        if len(d) == 0:
            return self.kialo.random_chain()[0]

        # Build claim pool + BM25 index if needed
        if self.kind not in self.kialo.claims:
            if self.kind == 'all':
                self.kialo.claims[self.kind] = list(self.kialo.parents)
            elif self.kind == 'has_cons':
                self.kialo.claims[self.kind] = [
                    c for c in self.kialo.parents if self.kialo.cons[c]
                ]
            elif self.kind == 'has_pros':
                self.kialo.claims[self.kind] = [
                    c for c in self.kialo.parents if self.kialo.pros[c]
                ]
            else:
                raise ValueError(f"Unknown claim kind: {self.kind}")

            self.kialo.bm25[self.kind] = BM25_Index(
                self.kialo.claims[self.kind],
                tokenizer=self.kialo.tokenizer
            )

        # ---------- speaker-aware + recency-weighted query ----------
        weighted_tokens = []

        USER_WEIGHT = 2.3
        BOT_WEIGHT = 1

        for i, turn in enumerate(d):
            base_weight = i + 1  # recency
            tokens = self.kialo.tokenizer(turn["content"])

            if turn.get("role") == "user":
                weight = base_weight * USER_WEIGHT
            else:
                weight = base_weight * BOT_WEIGHT

            weighted_tokens.extend(tokens * weight)

        bm25 = self.kialo.bm25[self.kind]
        candidates = bm25.get_top_n(
            weighted_tokens,
            self.kialo.claims[self.kind],
            n=3
        )

        # Fallback: last turn only
        if not candidates:
            log.info("Fallback to last turn only")
            last_tokens = self.kialo.tokenizer(d[-1]["content"])
            candidates = bm25.get_top_n(
                last_tokens,
                self.kialo.claims[self.kind],
                n=3
            )

        best_claim = random.choice(candidates)
        log.info(
            f"[black on bright_green]Chose similar claim from Kialo:\n{best_claim}[/black on bright_green]"
        )

        # Prefer counterarguments
        if best_claim in self.kialo.cons and self.kialo.cons[best_claim]:
            return random.choice(self.kialo.cons[best_claim])

        return best_claim


akiki = AkikiBot("Akiki", Kialo(glob.glob("data/*.txt")))


