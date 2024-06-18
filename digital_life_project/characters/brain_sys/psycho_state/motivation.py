"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: motivation.py
Description: Define the motivation class of SocioMind
"""
import numpy as np
from digital_life_project.characters.llm_api.gpt_prompt_brain import get_embedding


class Motivation:
    def __init__(self, self_name='', long_term: str='', short_term: str='', plot_id: int =-1, round: int =-1, time=None):
        self.self_name = self_name
        self.long_term = long_term
        self.short_term = short_term
        self.plot_id = plot_id
        self.round = round
        self.time = time
        self.embedding = None
        if self.embedding is None:
            self.set_motivation_embedding()
        
    
    def get_full_motivation(self):
        return f"<long_term>{self.long_term}<short_term>{self.short_term}"
    
    def get_full_description(self):
        return f"<self_name>{self.self_name}<long_term>{self.long_term}<short_term>{self.short_term}<plot_id>{self.plot_id}<round>{self.round}<time>{self.time}"
    
    def get_log_description(self):
        return f"<self_name>{self.self_name}<long_term>{self.long_term}<short_term>{self.short_term}<plot_id>{self.plot_id}"
    
    def get_current_motivation(self):
        return self.short_term
    
    def get_prompt_motivation(self):
        ### Paper: Robin R Vallacher and Daniel M Wegner. What do people think they’re doing? action identification and human behavior. Psychological review, 1987.
        prompt = f"{self.self_name}'s long-term motivation is [{self.long_term}] ;" +\
            f"{self.self_name}'s short-term motivation is [{self.short_term}]."
        return prompt
    
    def get_prompt_motivation_persona(self):
        prompt = f"{self.self_name}'s long-term motivation is [{self.long_term}] ;"
        return prompt
    
    def get_dicts(self):
        return {
            'self_name': self.self_name,
            'long_term': self.long_term,
            'short_term': self.short_term,
            'plot_id': self.plot_id,
            'round': self.round,
            'time': self.time,
        }
    
    def set_motivation_embedding(self):
        self.embedding = np.array(get_embedding(self.get_prompt_motivation()))
        return self.embedding
    

    
    
    