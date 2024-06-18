"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: emotion.py
Description: Define the emotion class of SocioMind
"""
import numpy as np
from digital_life_project.characters.llm_api.gpt_prompt_brain import get_embedding


class Emotion:
    def __init__(self, self_name='', description: str='', pleasure: int =-1, arousal: int =-1, 
                 dominance: int =-1, time=None, partner_name='', plot_id: int =-1, round: int =-1):
        self.self_name = self_name
        self.description = description
        self.pleasure = pleasure
        self.arousal = arousal
        self.dominance = dominance
        self.time = time
        self.partner_name = partner_name
        self.plot_id = plot_id
        self.round = round
        self.set_emotion_embedding()
    
    @staticmethod
    def parse_emotion_from_dicts(dicts):
        tmp_dicts = {}
        keys = ["pleasure", "arousal", "dominance"]
        for key in keys:
            tmp_dicts[key] = dicts[key]
        return Emotion(**tmp_dicts)
    
    def get_quantitative_description(self):
        return f"<pleasure>{self.pleasure}<arousal>{self.arousal}<dominance>{self.dominance}"
    
    def get_full_description(self):
        return f"<time>{self.time}<self_name>{self.self_name}<pleasure>{self.pleasure}<arousal>{self.arousal}<dominance>" + \
                f"{self.dominance}<description>{self.description}<partner_name>{self.partner_name}<plot_id>{self.plot_id}<round>{self.round}"
    
    def get_log_description(self):
        return f"<self_name>{self.self_name}<pleasure>{self.pleasure}<arousal>{self.arousal}<dominance>" + \
                f"{self.dominance}<description>{self.description}<partner_name>{self.partner_name}<plot_id>{self.plot_id}"
    
    def get_dicts(self):
        return {
            'time': self.time,
            'self_name': self.self_name,
            'partner_name': self.partner_name, 
            'pleasure': self.pleasure,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'description': self.description,
            'plot_id': self.plot_id,
            'round': self.round,
        }
    
    def get_prompt_desc_and_quantitative_emotion(self):
        ### Paper: Albert Mehrabian. Basic dimensions for a general psychological theory: Implications for personality, social, environmental, and developmental studies. 1980
        prompt = f"The emotion of {self.self_name} is described as [{self.description}]."
        prompt += f"According to PAD theory in a Likert scale with range (1-9), "+\
        f"{self.self_name}'s emotion is {self.pleasure if self.pleasure != -1 else 'unknown'} pleasure, " +\
            f"{self.arousal if self.arousal != -1 else 'unknown'} arousal, {self.dominance  if self.dominance != -1 else 'unknown'} dominance."
        return prompt
    
    def set_emotion_embedding(self):
        self.embedding = np.array(get_embedding(self.description))
        return self.embedding
    
    
