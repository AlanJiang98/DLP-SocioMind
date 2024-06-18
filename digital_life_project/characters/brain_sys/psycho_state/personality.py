"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: personality.py
Description: Define the personality class of SocioMind
"""
import numpy as np
from digital_life_project.characters.llm_api.gpt_prompt_brain import get_embedding


class Personality:
    def __init__(self, self_name='', description: str='', openness: int =-1, conscientiousness: int =-1, extraversion: int =-1,
                 agreeableness: int =-1, neuroticism: int =-1, time=None, plot_id: int =-1, round: int =-1):
        self.self_name = self_name
        self.description = description
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        self.time = time
        self.plot_id = plot_id
        self.round = round
        self.embedding = None
        self.set_personality_embedding()
        
    def get_quantitative_description(self):
        return f"<openness>{self.openness}<conscientiousness>{self.conscientiousness}<extraversion>{self.extraversion}<agreeableness>{self.agreeableness}<neuroticism>{self.neuroticism}"
    
    def get_qualitative_description(self):
        return self.description

    def get_full_description(self):
        return f"<time>{self.time}<self_name>{self.self_name}<openness>{self.openness}<conscientiousness>{self.conscientiousness}" + \
            f"<extraversion>{self.extraversion}<agreeableness>{self.agreeableness}<neuroticism>{self.neuroticism}<description>{self.description}" + \
                f"plot_id>{self.plot_id}<round>{self.round}"
    
    def set_personality_embedding(self):
        if self.description == '':
            desc = self.changing_big_five_to_description()
        else:
            desc = self.description
        self.embedding = np.array(get_embedding(desc))
        return self.embedding
        
    
    def get_dicts(self):
        return {
            'time': self.time,
            'self_name': self.self_name,
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
            'description': self.description,
            'plot_id': self.plot_id,
            'round': self.round,
        }
    
    @staticmethod
    def parse_personality_from_dicts(dicts):
        tmp_dicts = {}
        keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'description']
        for key in keys:
            if key in dicts.keys():
                tmp_dicts[key] = dicts[key]
        return Personality(**tmp_dicts)
    
    
    def get_prompt_desc_and_quantitative_personality(self):
        ### Paper: Oliver P John, Sanjay Srivastava, et al. The big-five trait taxonomy: History, measurement, and theoretical perspectives. 1999.
        prompt = f"{self.self_name} is a person described as {self.description}."
        prompt += f"According to Big Five Trait theory in a Likert scale with range (1-9), "+\
        f"{self.self_name}'s personality is {self.openness  if self.openness != -1 else 'unknown'} openness, " +\
            f"{self.conscientiousness if self.conscientiousness != -1 else 'unknown'} conscientiousness, {self.extraversion if self.extraversion != -1 else 'unknown'} extraversion," +\
                f" {self.agreeableness if self.agreeableness != -1 else 'unknown'} agreeableness, {self.neuroticism if self.neuroticism != -1 else 'unknown'} neuroticism."
        return prompt

    def get_prompt_desc_persona(self):
        prompt = f"{self.self_name} is a person described as {self.description}."
        return prompt
    
    def changing_big_five_to_description(self, big_five: dict=None):
        """
        changing big five to language description via Goldberg's personality trait markers
        """        
        Goldberg_personality_trait_markers = {
        "extraversion": {"E1 - Friendliness":["unfriendly",  "friendly"],
            "E2-0 - Gregariousness": ["introverted", "extraverted"],
            "E2-1 - Gregariousness": ["silent", "talkative"],
            "E3-0 - Assertiveness":["timid", "bold"],
            "E3-1 - Assertiveness": ["unassertive", "assertive"],
            "E4 - Activity Level": ["inactive", "active"],
            "E5-0 - Excitement-Seeking": ["unenergetic", "energetic"],
            "E5-1 - Excitement-Seeking": ["unadventurous", "adventurous and daring"],
            "E6 - Cheerfulness": ["gloomy", "cheerful"]},
        "agreeableness":{ "A1 - Trust":["distrustful", "trustful"],
            "A2-0 - Morality":["immoral", "moral"],
            "A2-1 - Morality":["dishonest", "honest"],
            "A3-0 - Altruism":["unkind", "kind"],
            "A3-1 - Altruism":["stingy", "generous"],
            "A3-2 - Altruism":["unaltruistic", "altruistic"],
            "A4 - Cooperation":["uncooperative", "cooperative"],
            "A5 - Modesty":["self-important", "humble"],
            "A6 - Sympathy":["unsympathetic", "sympathetic"],
            "AGR-0":["selfish", "unselfish"],
            "AGR-1":["disagreeable", "agreeable"]},
        "conscientiousness": {"C1 - Self-Efficacy":["unsure", "self-efficacious"],
            "C2 - Orderliness":["messy", "orderly"],
            "C3 - Dutifulness":["irresponsible", "responsible"],
            "C4 - Achievement-Striving":["lazy", "hardworking"],
            "C5 - Self-Discipline":["undisciplined", "self-disciplined"],
            "C6-0 - Cautiousness":["impractical", "practical"],
            "C-1 - Cautiousness":["extravagant", "thrifty"],
            "CON-0":["disorganized", "organized"],
            "CON-1":["negligent", "conscientious"],
            "CON-2":["careless", "thorough"],},
        "neuroticism": {"N1 - Anxiety":["relaxed", "tense"],
            "N1 - Anxiety":["at ease", "nervous"],
            "N1 - Anxiety":["easygoing", "anxious"],
            "N2 - Anger":["calm", "angry"],
            "N2 - Anger":["patient", "irritable"],
            "N3 - Depression":["happy", "depressed"],
            "N4 - Self-Consciousness":["unselfconscious", "self-conscious"],
            "N5 - Immoderation":["level-headed", "impulsive"],
            "N6 - Vulnerability":["contented", "discontented"],
            "N6 - Vulnerability":["emotionally stable", "emotionally unstable"],},
        "openness": {"O1 - Imagination":["unimaginative", "imaginative"],
            "O2-0 - Artistic Interests":["uncreative", "creative"],
            "O2-1 - Artistic Interests":["artistically unappreciative", "artistically appreciative"],
            "O2-2 - Artistic Interests":["unaesthetic", "aesthetic"],
            "O3-0 - Emotionality":["unreflective", "reflective"],
            "O3-1 - Emotionality":["emotionally closed", "emotionally aware"],
            "O4-0 - Adventurousness":["uninquisitive", "curious"],
            "O4-1 - Adventurousness":["predictable", "spontaneous"],
            "O5-0 - Intellect":["unintelligent", "intelligent"],
            "O5-1 - Intellect":["unanalytical", "analytical"],
            "O5-2 - Intellect":["unsophisticated", "sophisticated"],
            "O6 - Liberalism":["socially conservative", "socially progressive"],}}
        
        def get_template(score, adjectives: list):
            score = int(score)
            if score == 1:
                return f"extremely {adjectives[0]}"
            elif score == 2:
                return f"very {adjectives[0]}"
            elif score == 3:
                return f"{adjectives[0]}"
            elif score == 4:
                return f"a bit {adjectives[0]}"
            elif score == 5:
                return f"neither {adjectives[0]} nor {adjectives[1]}"
            elif score == 6:
                return f"a bit {adjectives[1]}"
            elif score == 7:
                return f"{adjectives[1]}"
            elif score == 8:
                return f"very {adjectives[1]}"
            elif score == 9:
                return f"extremely {adjectives[1]}"
            else:
                return f"neutral"
        
        if big_five is None:
            big_five = self.get_dicts()
        
        personality_desc = f"{self.self_name} is a person described as [ "
        
        for key in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            if key in big_five.keys():
                score = big_five[key]
                for subkey in Goldberg_personality_trait_markers[key].keys():
                    adjectives = Goldberg_personality_trait_markers[key][subkey]
                    personality_desc += get_template(score, adjectives) + ", "
        personality_desc = personality_desc[:-2] + "]."
        return personality_desc