"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: psychostate.py
Description: Psychological state of the SocioMind
"""

import datetime
from digital_life_project.characters.brain_sys.psycho_state.emotion import Emotion
from digital_life_project.characters.brain_sys.psycho_state.personality import Personality
from digital_life_project.characters.brain_sys.psycho_state.motivation import Motivation
from digital_life_project.characters.brain_sys.psycho_state.core_self import Coreself
from digital_life_project.characters.brain_sys.psycho_state.behavior import Behavior
from digital_life_project.characters.brain_sys.psycho_state.plot import Plot, Topic
from digital_life_project.characters.llm_api.gpt_prompt_brain import *
from digital_life_project.characters.brain_sys.memory_modules.episodic_semantic_memory import Event
from digital_life_project.characters.brain_sys.psycho_state.persona_instruct import PersonaInstructionDatabase



class PsychoState():
    def __init__(self, name, config, character):
        self.name = name
        self.config = config
        self.character = character
        
        self.persona_instruction_database = PersonaInstructionDatabase(config)

        self.current_plot_id = -1
        self.current_round = -1
        
        self.personality_list = []
        self.emotion_list = []
        self.motivation_list = []
        self.core_self_list = []
        self.current_topic_list = []
        self.used_topic_list = []
        self.topics_for_current_plot = []
        self.current_behavior = None
        self.perserved_observed_info = {}
        self.perceived_behavior = []
        self.proposed_plot_list = []
        self.current_plot_config = {}
        
        self.plot_state = 'plot_finished' # plan, working, plot_finished, quit
        
        # generate & regenerate the topic
        # ranking with emergency and importance
        self.init_personality(self.config['characters_info'][self.name]['personality'], plot_id=0, round=0)
        self.init_emotion(self.config['characters_info'][self.name]['emotion'], plot_id=0, round=0, partner_name=None)
        self.init_motivation(self.config['characters_info'][self.name]['motivation'], plot_id=0, round=0)
        self.init_core_self(self.config['characters_info'][self.name]['core_self'], plot_id=0, round=0)
    
    
    def save_current_plot_state(self, save_dir=''):
        saved_attributes = {}
        for name in ['current_plot_id', 'current_round', 'personality_list', 'emotion_list', 'motivation_list', 'core_self_list', 'current_topic_list', 'used_topic_list', 'plot_state']:
            saved_attributes[name] = getattr(self, name)
        return saved_attributes
    
    def load_current_plot_state(self, saved_attributes):
        for name in ['current_plot_id', 'current_round', 'personality_list', 'emotion_list', 'motivation_list', 'core_self_list', 'current_topic_list', 'used_topic_list', 'plot_state']:
            setattr(self, name, saved_attributes[name])

    # can be simplied by using a decorator 
    # initialize the personality, emotion, motivation, self
    def init_personality(self, dicts, plot_id=0, round=0):
        if dicts['description'] == '':
            self.personality_list[0:0] = [self.get_personality_by_quantitative(dicts)]
        elif dicts['openness'] not in list(range(1, 10)):
            self.personality_list[0:0] = [self.get_personality_by_description(dicts['description'])]
        else:
            personality = Personality(**dicts)
            personality.plot_id = plot_id
            personality.round = round
            personality.self_name = self.name
            personality.time = datetime.datetime.now()
            self.personality_list[0:0] = [personality]
    
    def init_emotion(self, dicts, plot_id=0, round=0, partner_name=''):
        emotion = Emotion(**dicts)
        emotion.plot_id = plot_id
        emotion.round = round
        emotion.self_name = self.name
        emotion.time = datetime.datetime.now()
        emotion.partner_name = partner_name
        self.emotion_list[0:0] = [emotion]
        
    
    def init_motivation(self, dicts, plot_id=0, round=0):
        motivation = Motivation(**dicts)
        motivation.plot_id = plot_id
        motivation.round = round
        motivation.self_name = self.name
        motivation.time = datetime.datetime.now()
        self.motivation_list[0:0] = [motivation]
    
    def init_core_self(self, dicts, plot_id=0, round=0):
        core_self = Coreself(**dicts)
        core_self.plot_id = plot_id
        core_self.round = round
        core_self.self_name = self.name
        core_self.time = datetime.datetime.now()
        self.core_self_list[0:0] = [core_self]
        
    # controlable by the user
    def get_emotion_by_description(self, description):
        """
        transfer the description to quantitative emotion
        """
        quant_res = run_gpt_get_quantitative_emotion_from_description(description)
        emotion = Emotion.parse_emotion_from_dicts(quant_res)
        emotion.description = description
        emotion.time = datetime.datetime.now()
        emotion.self_name = self.name
        emotion.partner_name = self.character.partners[0].name
        emotion.plot_id = self.current_plot_id
        emotion.round = self.current_round
        return emotion

    def get_emotion_by_quantitative(self, quantitative):
        """
        transfer the quantitative emotion to description
        """
        desc_res = run_gpt_get_qualitative_emotion_from_quantitative(quantitative)
        emotion = Emotion.parse_emotion_from_dicts(quantitative)
        emotion.description = desc_res
        emotion.set_emotion_embedding()
        emotion.time = datetime.datetime.now()
        emotion.self_name = self.name
        emotion.partner_name = self.character.partners[0].name
        emotion.plot_id = self.current_plot_id
        emotion.round = self.current_round
        return emotion
    
    def get_personality_by_description(self, description):
        """
        transfer the description to quantitative personality
        """
        quant_res = run_gpt_get_quantitative_personality_from_description(description)
        personality = Personality.parse_personality_from_dicts(quant_res)
        personality.description = description
        personality.time = datetime.datetime.now()
        personality.self_name = self.name
        personality.plot_id = self.current_plot_id
        personality.round = self.current_round
        return personality
    
    
    def get_personality_by_quantitative(self, quantitative):
        """
        transfer the quantitative personality to description
        """
        quantitative.pop('description')
        quant_res = run_gpt_get_qualitative_personality_from_quantitative(quantitative)
        personality = Personality.parse_personality_from_dicts(quantitative)
        personality.description = quant_res
        personality.set_personality_embedding()
        personality.time = datetime.datetime.now()
        personality.self_name = self.name
        personality.plot_id = self.current_plot_id
        personality.round = self.current_round
        return personality
        
    
    def start_new_plot_setup(self, plot_config):
        self.current_plot_id += 1
        self.current_round = 0
        if 'summary' not in plot_config:
            plot_config['summary'] = run_gpt_summarize_sentence(plot_config['plot_background'])
        new_plot = Plot(self.name, self.current_plot_id, 
                        time=datetime.datetime.now(), plot_background=plot_config['plot_background'],
                        summary=plot_config['summary'])
        
        if 'behavior' in plot_config:
            behavior_dicts = {}
            for key in plot_config['behavior'].keys():
                if key in Behavior.get_keywords():
                    behavior_dicts[key] = plot_config['behavior'][key]
            
            behavior_dicts['self_name'] = self.name
            behavior_dicts['time'] = datetime.datetime.now()
            behavior_dicts['round'] = self.current_round
            behavior_dicts['plot_id'] = self.current_plot_id
            behavior_dicts['partner_name'] = self.character.partners[0].name
            behavior = Behavior(**behavior_dicts)
        else:
            behavior = Behavior(self_name=self.name, time=datetime.datetime.now(), round=self.current_round, \
                plot_id=self.current_plot_id, partner_name=self.character.partners[0].name)
        behavior.set_embedding()
        self.current_behavior = behavior
        
        if 'emotion' in plot_config.keys():
            if type(plot_config['emotion']) is str:
                emotion = self.get_emotion_by_description(plot_config['emotion'])
            elif type(plot_config['emotion']) is dict:
                emotion = self.get_emotion_by_quantitative(plot_config['emotion'])
            self.emotion_list[0:0] = [emotion]
        
        return new_plot, behavior

    def generate_topics_with_events_and_personal_info(self, prompt_personal_info, prompt_relevant_memory):
        used_topics = [topic.get_topic_prompt() for topic in self.used_topic_list[:self.config['max_used_topic_retrieval']:] if topic.used]
        prompt_relevant_memory += f"Previous used topics are: {used_topics}\n"
           
        topic_dict_lists = run_gpt_get_topics_from_events_personal_info(prompt_personal_info, prompt_relevant_memory)
        # previous topics needed
        for topic_dict in topic_dict_lists:
            topic = Topic.get_topic_from_dicts(topic_dict)
            topic.time = datetime.datetime.now()
            topic.self_name = self.name
            topic.created_plot_id = self.current_plot_id
            topic.round = self.current_round
            self.current_topic_list[0:0] = [topic]
        self.current_topic_list = sorted(self.current_topic_list, key=lambda x: x.get_topic_score(), reverse=True)  

    
    def deduplicate_topics(self):
        # sort the topics by score
        if self.current_topic_list == []:
            return
        self.current_topic_list = sorted(self.current_topic_list, key=lambda x: x.get_topic_score(), reverse=True)
        
        if len(self.current_topic_list) > self.config['max_topic_cache']:
            self.current_topic_list = self.current_topic_list[:self.config['max_topic_cache']]
        
        # remove the duplicated topics
        prompt = f"Below are the proposed topics between two characters:\n" +\
            f"Each topic has an ID and description with poignancy and emergency in a python dict format.\n" +\
                f"Due to potential duplicates or similarities among these topics, please select the IDs of the topics after removing the duplicates.\n"
        
        prompt += f"The topics are as follows:\n"
        for i, topic in enumerate(self.current_topic_list):
            prompt += f"Topic {i}: {topic.get_topic_prompt()}\n"
        prompt += "The given ids should be in a python list format, like list indices.\n"
        
        indices = run_gpt_get_indices_from_deduplicating_topics(prompt, topic_acc=len(self.current_topic_list), verbose=False)
        self.current_topic_list = [self.current_topic_list[i] for i in sorted(indices)]
    
    def emotion_update_process(self, innate_trait_prompt, memory_prompt, behavior_desc, previous_emotion):
        prompt = innate_trait_prompt + memory_prompt
        # if self.emotion_list != [] and self.emotion_list[0].plot_id == self.current_plot_id:
        #     prompt += f"\n----\nHer/His previous emotion is {self.emotion_list[0].get_quantitative_description()}\n"
        prompt += f"\n----\nNow her/his current behavior is {behavior_desc}\n" +\
            f"According to Paul Ekman's basic emotion theory, human have basic emtions: wrath, grossness, fear, joy, loneliness, shock, " +\
                f"amusement, contempt, contentment, embarrassment, excitement, guilt, pride in achievement, relief, satisfaction, sensory pleasure, and shame." +\
                    f"Before these behaviors, his/her emotion are: {previous_emotion}\n" +\
                        f"What is her/his current emotion? You'd better give a change. The change of emotion should reflect the previous behaviors.\n" +\
            f"So based on the personality, previous emotion, motivation, relationship, and relevant memories, using the psychological theory, her/his current emotion is:\n"
        emotion_dicts = run_gpt_get_emotion_from_prompt(prompt)
        updated_emotion = Emotion(**emotion_dicts)
        updated_emotion.time = datetime.datetime.now()
        updated_emotion.self_name = self.name
        updated_emotion.partner_name = self.character.partners[0].name
        updated_emotion.plot_id = self.current_plot_id
        updated_emotion.round = self.current_round
        self.emotion_list[0:0] = [updated_emotion]
        return updated_emotion
    
    
        