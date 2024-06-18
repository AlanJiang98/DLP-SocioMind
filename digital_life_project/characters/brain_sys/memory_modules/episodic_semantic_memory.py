"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: episodic_semantic_memory.py
Description: Define the episodic & semantic memory class for the characters
"""
import numpy as np
from digital_life_project.characters.llm_api.gpt_prompt_brain import get_embedding, run_gpt_get_keywords_poignancy_from_description


class Event:
    def __init__(self, self_name='', description: str='', keywords: list=[], poignancy=5, emergency=5, embedding=None, time=None, plot_id: int =-1, round: int =-1):
        self.self_name = self_name
        self.description = description
        self.time = time
        self.plot_id = plot_id
        self.round = round
        self.keywords = keywords
        self.embedding = embedding
        self.poignancy = poignancy
        self.emergency = emergency
        self.access_times = 1
        self.forgot = False
        if self.embedding is None:
            self.embedding = np.array(get_embedding(self.description)) 
    
    def set_keywords_from_descriptions(self, background: str=''):
        # process the description from human instructions
        output = run_gpt_get_keywords_poignancy_from_description(self.description, background=background)
        if output is not False:
            self.poignancy = output['poignancy']
            self.emergency = output['emergency']
            self.keywords = output['keywords']
    
    def get_forgetting_rate(self, plot_id, a=0.4, importance_base=3, k_per_plot=4, threshold=0.6):
        ### using Ebbinghaus forgetting curve
        ### Paper: Averell, L., & Heathcote, A. (2011). The form of the forgetting curve and the fate of memories. Journal of mathematical psychology, 55(1), 25-35.
        rate = a + (1-a) * np.exp(- k_per_plot * (plot_id - self.plot_id) / (np.exp2(self.access_times) *(importance_base + self.poignancy)))
        if rate < threshold:
            self.forgot = True
            return 0.0
        else:
            return rate
    
    def get_score_from_embedding(self, embedding, plot_id, a=0.1, importance_base=3, k_per_plot=4, threshold=0.3):
        similarity = np.dot(self.embedding, embedding)
        forget_rate = self.get_forgetting_rate(plot_id, a=a, importance_base=importance_base, k_per_plot=k_per_plot, threshold=threshold)
        return similarity * forget_rate

    def get_desc_and_poignancy(self):
        return f"Event: [{self.description}], poignancy: {self.poignancy}, emergency: {self.emergency} time: {self.time}.\n"
    
    def get_dicts(self):
        return {
            'self_name': self.self_name,
            'description': self.description,
            'time': self.time,
            'plot_id': self.plot_id,
            'round': self.round,
            'keywords': self.keywords,
            'poignancy': self.poignancy,
            'access_times': self.access_times,
            'forgot': self.forgot,
            'emergency': self.emergency,
        }
    
    def get_full_description(self):
        desc = ""
        for key in ['self_name', 'description', 'time', 'plot_id', 'round', 'keywords', 'poignancy', 'emergency', 'access_times', 'forgot']:
            desc += f"<{key}>{self.__dict__[key]}"
        return desc
    
    def get_log_description(self):
        desc = ""
        for key in ['self_name', 'description', 'plot_id', 'keywords', 'poignancy', 'emergency']:
            desc += f"<{key}>{self.__dict__[key]}"
        return desc
    
        
class Thought:
    def __init__(self, self_name='', description: str='', keywords=[], poignancy=5, embedding=None, time=None, plot_id: int =-1):
        self.self_name = self_name
        self.description = description
        self.time = time
        self.embedding = embedding
        self.poignancy = poignancy
        self.plot_id = plot_id
        self.access_times = 1
        self.keywords = keywords
        self.forgot = False
        
    def set_embedding_from_description(self):
        self.embedding = np.array(get_embedding(self.description))
        
    def get_forgetting_rate(self, plot_id, a=0.1, importance_base=3, k_per_plot=2, threshold=0.3):
        ### using Ebbinghaus forgetting curve
        ### Paper: Averell, L., & Heathcote, A. (2011). The form of the forgetting curve and the fate of memories. Journal of mathematical psychology, 55(1), 25-35.
        rate = a + (1-a) * np.exp(- k_per_plot * (plot_id - self.plot_id) / (np.exp2(self.access_times) *(importance_base + self.poignancy)))
        if rate < threshold:
            self.forgot = True
        else:
            return rate
    
    def get_score_from_embedding(self, embedding, plot_id, a=0.4, importance_base=3, k_per_plot=2, threshold=0.6):
        similarity = np.dot(self.embedding, embedding)
        forget_rate = self.get_forgetting_rate(plot_id, a=a, importance_base=importance_base, k_per_plot=k_per_plot, threshold=threshold)
        return similarity * forget_rate

    def mark_accessed(self):
        self.access_times += 1

    def get_desc_and_poignancy(self):
        return f"Thought: [{self.description}], poignancy: {self.poignancy}, time: {self.time}.\n"
    
    def get_dicts(self):
        return {
            'self_name': self.self_name,
            'description': self.description,
            'time': self.time,
            'plot_id': self.plot_id,
            'poignancy': self.poignancy,
            'access_times': self.access_times,
            'forgot': self.forgot,
            'keywords': self.keywords,
        }
    
    def get_full_description(self):
        desc = ""
        for key in ['self_name', 'description', 'time', 'plot_id', 'poignancy', 'access_times', 'forgot', 'keywords']:
            desc += f"<{key}>{self.__dict__[key]}"
        return desc

    def get_log_description(self):
        desc = ""
        for key in ['self_name', 'description', 'plot_id', 'poignancy', 'keywords']:
            desc += f"<{key}>{self.__dict__[key]}"
        return desc