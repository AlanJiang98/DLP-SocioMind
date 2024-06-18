"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: social_memory.py
Description: Define the social relationship class for the characters
"""
from digital_life_project.characters.llm_api.gpt_prompt_brain import *
import numpy as np


class Relationship:
    def __init__(self, self_name: str='', description: str='', attitude: str='', intimacy: int=-1, trust: int=-1, supportiveness: int=-1, time=None, partner_name: str='', plot_id: int =-1, round: int =-1):
        self.self_name = self_name
        self.desc = description
        self.intimacy = intimacy
        self.trust = trust
        self.supportiveness = supportiveness
        self.time = time
        self.partner_name = partner_name
        self.attitude = attitude
        self.plot_id = plot_id
        self.round = round
        self.set_relationship_embedding()
    
    @staticmethod
    def parse_relationship_from_dicts(dicts):
        tmp_dicts = {}
        keys = ["intimacy", 'trust', "supportiveness", 'description', 'attitude']
        for key in keys:
            if key in dicts.keys():
                tmp_dicts[key] = dicts[key]
        return Relationship(**tmp_dicts)
    
    def get_key_description(self):
        return f"<intimacy>{self.intimacy}<trust>{self.trust}<supportiveness>{self.supportiveness}"
    
    def get_full_description(self):
        return f"<time>{self.time}<self_name>{self.self_name}<intimacy>{self.intimacy}<trust>{self.trust}" +\
            f"<supportiveness>{self.supportiveness}<description>{self.desc}<attitude>{self.attitude}<partner_name>{self.partner_name}" +\
                f"<plot_id>{self.plot_id}<round>{self.round}"

    def get_log_description(self):
        return f"<self_name>{self.self_name}<intimacy>{self.intimacy}<trust>{self.trust}" +\
            f"<supportiveness>{self.supportiveness}<description>{self.desc}<attitude>{self.attitude}<partner_name>{self.partner_name}" +\
                f"<plot_id>{self.plot_id}"
    
    def get_dicts(self):
        return {
            'time': self.time,
            'self_name': self.self_name,
            'partner_name': self.partner_name, 
            'intimacy': self.intimacy,
            'trust': self.trust,
            'supportiveness': self.supportiveness,
            'description': self.desc,
            'attitude': self.attitude,
            'plot_id': self.plot_id,
            'round': self.round
        }
    
    def get_prompt_relationship(self):
        prompt = f"The relationship between {self.self_name} and {self.partner_name} is [{self.desc}]." +\
            f"{self.self_name}'s attitude towards {self.partner_name} is [{self.attitude}]." +\
            f"According to social pyschological view to measure the social relationship in a Likert scale range (1-9), " +\
            f"the intimacy is {self.intimacy if self.intimacy != -1 else 'unknown'}, the trust is {self.trust if self.trust != -1 else 'unknown'}, and the supportiveness is {self.supportiveness if self.supportiveness != -1 else 'unknown'}."
        return prompt

    def get_prompt_relationship_persona(self):
        prompt = f"The relationship between {self.self_name} and {self.partner_name} is [{self.desc}]."
        return prompt

    def get_quantitative_relationship_from_description(self):
        quantitative = run_gpt_get_quantitative_relationship_from_description(self.desc)
        for key in quantitative.keys():
            setattr(self, key, quantitative[key])
    
    def set_relationship_embedding(self):
        desc = f"The relationship between {self.self_name} and {self.partner_name} is [{self.desc}]." +\
            f"{self.self_name}'s attitude towards {self.partner_name} is [{self.attitude}]."
        self.embedding = np.array(get_embedding(desc))
        return self.embedding