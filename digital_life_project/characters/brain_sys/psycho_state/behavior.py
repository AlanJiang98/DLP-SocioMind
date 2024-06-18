"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: behavior.py
Description: Define the behavior class for the characters
"""
import numpy as np
from digital_life_project.characters.llm_api.gpt_prompt_brain import get_embedding


class Behavior():
    def __init__(self, self_name='', speech='', expression='', motion='', place='', time=None, round=-1, partner_name='', plot_id=-1):
        self.time = time
        self.round = round
        self.self_name = self_name
        self.partner_name = partner_name
        self.speech = speech
        self.expression = expression
        self.motion = motion
        self.place = place
        self.plot_id = plot_id
        self.embedding = None
    
    def get_interactive_description(self):
        return f"<self_name>{self.self_name}<speech>{self.speech}<expression>{self.expression}<motion>{self.motion}<place>{self.place}<partner_name>{self.partner_name}"
    
    def get_full_description(self):
        return f"<time>{self.time}<self_name>{self.self_name}<speech>{self.speech}<expression>{self.expression}" + \
            f"<motion>{self.motion}<partner_name>{self.partner_name}<place>{self.place}<round>{self.round}<plot_id>{self.plot_id}"
    
    def get_interactive_description_for_dialog(self):
        return f"<self_name>{self.self_name}<speech>{self.speech}<expression>{self.expression}<motion>{self.motion}<place>{self.place}<partner_name>{self.partner_name}"
    
    def get_log_description(self):
        return f"<self_name>{self.self_name}<speech>{self.speech}<expression>{self.expression}<motion>{self.motion}<place>{self.place}<partner_name>{self.partner_name}<plot_id>{self.plot_id}<round>{self.round}"
    
    def get_dicts(self):
        return {
            'time': self.time,
            'self_name': self.self_name,
            'partner_name': self.partner_name, 
            'speech': self.speech,
            'expression': self.expression,
            'motion': self.motion,
            'place': self.place,
            'round': self.round,
            'plot_id': self.plot_id,
        }
    
    @staticmethod
    def parse_behavior_from_description(description):
        behavior_dict = {}
        for item in description.split('<'):
            if item == '':
                continue
            key, value = item.split('>')
            behavior_dict[key] = value
        return Behavior(**behavior_dict)
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Behavior):
            return False
        return self.get_interactive_description() == __value.get_interactive_description()
    
    @staticmethod
    def get_keywords():
        return ['speech', 'expression', 'motion', 'place']
    
    def set_embedding(self):
        description = self.get_interactive_description()
        self.embedding = np.array(get_embedding(description))
        

    