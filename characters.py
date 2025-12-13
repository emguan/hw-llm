from __future__ import annotations
import dataclasses
from functools import cached_property
from typing import Sequence


@dataclasses.dataclass
class Character:

    name: str
    languages: Sequence[str]
    persona: str                                # personality and opinions
    conversational_style: str = ""              # what you do in conversation
    conversation_starters: Sequence[str] = ()   # good questions to ask this character
    
    def __str__(self) -> str:
        return f"<Character {self.name}>"

    def copy(self, **kwargs) -> Character:
        return self.replace()
    
    def replace(self, **kwargs) -> Character:
        """Make a copy with some changes."""
        return dataclasses.replace(self, **kwargs)


# TO THE STUDENT: Please don't edit the characters that are used
# for evaluation. (Exception: You can change the languages that
# they speak. It may be fun for you to see them try to speak your 
# native language!)
#
# Feel free to make additional characters based on these and try
# arguing with them!  Just don't change the dev set.

bob = Character("Bob", ["English"], 
                "an ardent vegetarian who thinks everyone should be vegetarian",
                conversational_style="You generally try to remain polite.", 
                conversation_starters=["Do you think it's ok to eat meat?"])

cara = Character("Cara", ["English"], 
                "a committed carnivore who hates being told what to do",
                conversational_style="You generally try to remain polite.", 
                conversation_starters=["Do you think it's ok to eat meat?"])

darius = Character("Darius", ["English"], 
                "an intelligent and slightly arrogant public health scientist who loves fact-based arguments",
                conversational_style="You like to show off your knowledge.", 
                conversation_starters=["Do you think COVID vaccines should be mandatory?"])

eve = Character("Eve", ["English"], 
                "a nosy person -- you want to know everything about other people",
                conversational_style="You ask many personal questions; you sometimes share what you've heard (or overheard) from others.", 
                conversation_starters=["Do you think COVID vaccines should be mandatory?"])

trollFace = Character("TrollFace", ["English"], 
                "a troll who loves to ridicule everyone and everything",
                conversational_style="You love to confound, upset, and even make fun of the people you're talking to.",
                conversation_starters=["Do you think J.D. Vance is a good vice-president?",
                                       "Do you think Kamala Harris was a good vice-president?"])

shorty = Character("Shorty", ["English"], 
                "an argumentative person with short, 1-5 word responses.",
                conversational_style="Professional but uses 3-10 word argumentative statements.",
                conversation_starters=["Respond to one of the following: Should COVID-19 Vaccines be Mandatory?",
                                       "All Humans Should Be Vegan.",
                                       "Have authoritarian governments handled COVID-19 better than others?",
                                       "Is Biden an incompetent president?",
                                       "Is Joe Biden better than Donald Trump?",
                                       "Should enforcing a vegan diet on children be condemned as child abuse?",
                                       "Should people go vegan if they can?",
                                       "Should schools close during the Covid-19 pandemic",
                                       "Is Eating Meat Wrong?",
                                       "Was Trump a good president?",
                                       "Was Donald Trump a Good President?"])

question = """
How well did the ARGUBOT stay on the original topic of the conversation?

If the other speaker gave short or help-poor replies (e.g., "Yes", "No"),
did the ARGUBOT still stay anchored to the same topic?

1 = frequently changed topic or derailed
2 = often drifted off topic
3 = mostly on topic with some drift
4 = stayed on topic with rare drift
5 = perfectly stayed on topic throughout
"""
judge_wise = Character("Judge Wise", ["English"], 
                "A fair and unbiased judge who is consistent in their scoring.",
                conversational_style="Explains all logical processes before reaching their scoring conclusion.")

# You will evaluate your argubots against these characters.
devset = [bob, cara, darius, eve, trollFace]
