"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: plot.py
Description: Define the plot class for the characters
"""
class Topic:
    def __init__(self, self_name='', description: str='', poignancy: int =0, emergency: int = 0, time=None, summary="", created_plot_id: int =-1, round: int =-1, partner_name=''):
        self.self_name = self_name
        self.description = description
        self.poignancy = poignancy
        self.emergency = emergency
        self.time = time
        self.created_plot_id = created_plot_id
        self.round = round
        self.used = False
        self.used_plot_id = -1
        self.partner_name = partner_name
        self.summary = summary
    
    def get_topic_from_dicts(dicts):
        for key in dicts:
            if key not in ['self_name', 'description', 'poignancy', 'emergency', 'time', 'created_plot_id', 'round', 'partner_name', 'summary']:
                raise ValueError(f"Invalid key {key} for topic.")
        return Topic(**dicts)
    
    
    def get_topic_score(self, poignancy_weight=1, emergency_weight=2):
        if self.used:
            return 0
        else:
            return self.poignancy * poignancy_weight + self.emergency * emergency_weight

    def get_topic_prompt(self):
        return "{{'description': '{}', 'poignancy': {}, 'emergency': {}}}".format(self.description, self.poignancy, self.emergency)
    
    def get_used_topic_info(self):
        if self.used:
            return f"Topic: [{self.description}] in plot [{self.used_plot_id}] with partner [{self.partner_name}]].\n"
    
    def get_full_description(self):
        desc = ""
        for key in self.__dict__:
            desc += f"<{key}>{self.__dict__[key]}"
        return desc
    
    def get_log_description(self):
        desc = ""
        for key in ['self_name', 'description', 'poignancy', 'emergency',  'partner_name', 'summary']:
            desc += f"<{key}>{self.__dict__[key]}"
        return desc
    
    def get_dicts(self):
        return self.__dict__
    

class Plot():
    def __init__(self, self_name: str='', plot_id: int =-1, time=None, plot_background: str ='', summary='',
                 topics: list = [], behaviors: list = [], generated=False):
        self.self_name = self_name
        self.plot_id = plot_id
        self.time = time
        self.plot_background = plot_background
        self.topics = topics
        self.behaviors = behaviors
        self.generated = generated
        self.summary = summary