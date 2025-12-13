"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
from openai import OpenAI
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

        if best_claim in self.kialo.cons and self.kialo.cons[best_claim]:
            return random.choice(self.kialo.cons[best_claim])

        return best_claim


akiki = AkikiBot("Akiki", Kialo(glob.glob("data/*.txt")))

class RAGAgent(Agent):

    def __init__(self, name: str, kialo: Kialo, *, kind: str = "has_cons",
                 n_retrieved: int = 3, n_neighbors_each: int = 2,
                 temperature: float = 0.7):
        self.name = name
        self.kialo = kialo
        self.kind = kind
        self.n_retrieved = n_retrieved
        self.n_neighbors_each = n_neighbors_each

        self.query_llm = LLMAgent(
            name=f"{name}-Query",
            temperature=0,
            system=(
                "You rewrite the user's last message as an explicit, standalone claim.\n"
                "Rules:\n"
                "- Use the full dialogue for context.\n"
                "- Output 1–2 sentences.\n"
            ),
            speaker_names=False,
            compress=False
        )

        self.rag_llm = LLMAgent(
            name=name,
            temperature=temperature,
            system=(
                "You are Aragorn, an intelligent debate partner.\n"
                "Goal: respond to the user's last turn thoughtfully and stay on topic.\n"
                "You will be given:\n"
                "1) The dialogue so far\n"
                "2) A short EVIDENCE document with relevant claims and counterarguments\n\n"
                "Instructions:\n"
                "- Use the EVIDENCE to support your response, but do not quote it verbatim.\n"
                "- If the evidence is mixed, acknowledge uncertainty.\n"
                "- Reply in 1–2 sentences.\n"
                "- Be polite and specific.\n"
            ),
            speaker_names=False,
            compress=False
        )

    def _user_last_turn(self, d: Dialogue) -> str:
        return d[-1]["content"] if len(d) else ""

    def _make_query(self, d: Dialogue) -> str:
        prompt = (
            "Rewrite the user's last message as an explicit standalone claim.\n"
            "Output only the rewritten claim (no labels)."
        )
        return self.query_llm.ask_quietly(d, speaker="User", question=prompt).strip()

    def _retrieve_doc(self, query: str) -> str:
        neighbors = self.kialo.closest_claims(query, n=self.n_retrieved, kind=self.kind)
        if not neighbors:
            neighbors = self.kialo.closest_claims(query, n=self.n_retrieved, kind="all")

        lines = []
        lines.append("EVIDENCE (Kialo claims; treat as curated talking points):")

        for i, claim in enumerate(neighbors, start=1):
            pros = self.kialo.pros.get(claim, [])[: self.n_neighbors_each]
            cons = self.kialo.cons.get(claim, [])[: self.n_neighbors_each]

            lines.append(f"\n[{i}] Topic claim: {claim}")

            if pros:
                lines.append("  Pros:")
                for p in pros:
                    lines.append(f"   - {p}")
            if cons:
                lines.append("  Cons:")
                for c in cons:
                    lines.append(f"   - {c}")

        return "\n".join(lines)

    def response(self, d: Dialogue, **kwargs) -> str:
        if len(d) == 0:
            return self.kialo.random_chain()[0]

        explicit_claim = self._make_query(d)

        evidence_doc = self._retrieve_doc(explicit_claim)

        rag_prompt = (
            f"USER CLAIM (paraphrased): {explicit_claim}\n\n"
            f"{evidence_doc}\n\n"
            "Now write Aragorn's next reply to the user."
        )

        out = self.rag_llm.ask_quietly(d, speaker="User", question=rag_prompt, **kwargs)
        return out.strip()

aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")))

class AwsomAgent(Agent):
    def __init__(self, name: str, temperature: float = 0.6):
        self.name = name

        self.planner = LLMAgent(
            name=f"{name}-Planner",
            temperature=0,
            system=(
                "You analyze a debate turn.\n"
                "Given the dialogue so far, identify:\n"
                "1) The user's main claim or stance\n"
                "2) One effective way to challenge or broaden it\n\n"
                "Respond with a short plan (2–3 bullet points).\n"
            ),
            speaker_names=False,
            compress=False
        )

        self.speaker = LLMAgent(
            name=name,
            temperature=temperature,
            system=(
                "You are Awsom, an intelligent and confident debate partner.\n"
                "Goal: challenge the user's position thoughtfully and clearly.\n\n"
                "Instructions:\n"
                "- Take a clear position that pushes back on the user.\n"
                "- Stay focused on the topic.\n"
                "- Be respectful but not overly cautious.\n"
                "- Reply in 1–2 sentences.\n"
            ),
            speaker_names=False,
            compress=False
        )

    def response(self, d: Dialogue, **kwargs) -> str:
        if len(d) == 0:
            return "What’s an issue you feel strongly about?"

        plan = self.planner.ask_quietly(
            d,
            speaker="Analyst",
            question="Analyze the user's stance and plan a challenge."
        )

        prompt = (
            f"PRIVATE PLAN (do not reveal explicitly):\n{plan}\n\n"
            "Now write Awsom's reply to the user."
        )

        out = self.speaker.ask_quietly(
            d,
            speaker="User",
            question=prompt,
            **kwargs
        )

        return out.strip()

awsom = AwsomAgent("Awsom")