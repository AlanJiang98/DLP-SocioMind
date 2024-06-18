"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: sociomind.py
Description: SocioMind class for Autonomous Character in digital life project
"""

import datetime
import copy
from digital_life_project.characters.brain_sys.memory import Memory
from digital_life_project.characters.brain_sys.psychostate import PsychoState
from digital_life_project.characters.brain_sys.psycho_state.behavior import Behavior
from digital_life_project.characters.llm_api.gpt_prompt_brain import *
from digital_life_project.characters.brain_sys.memory_modules.episodic_semantic_memory import Event, Thought
from digital_life_project.characters.brain_sys.memory_modules.social_memory import Relationship
from digital_life_project.characters.brain_sys.psycho_state.motivation import Motivation
from digital_life_project.characters.brain_sys.psycho_state.plot import Plot, Topic
import ast
import csv
import numpy as np
from digital_life_project.characters.brain_sys.utils import *
import pickle
import yaml


class SocioMind():
    def __init__(self, name, config, character):
        self.name = name
        self.config = config
        self.psycho_state = PsychoState(name, config, character)
        self.memory = Memory(name, config)
        self.working_memory = {}
        self.character = character
        if self.config['replay_dir'] != '':
            self.load_replay()

    
    def perception(self):
        # only use one partner for one message here
        partner_name = self.character.partners[0].name
        behavior_desc = self.psycho_state.perserved_observed_info[partner_name]['behavior_desc']
        if behavior_desc.startswith('<self_name>'):
            p_behavior = Behavior.parse_behavior_from_description(behavior_desc)
            # obtain lastest behaviors
            latest_behaviors = self.memory.get_latest_behaviors(retention=self.config['retention_perception'])
            latest_behaviors = [latest_behavior.behavior for latest_behavior in latest_behaviors]
            if p_behavior not in latest_behaviors:
                p_behavior.time = datetime.datetime.now()
                p_behavior.round = self.psycho_state.current_round
                p_behavior.plot_id = self.psycho_state.current_plot_id
                p_behavior.set_embedding()
                self.memory.add_behavior(p_behavior)
                self.psycho_state.perceived_behavior[0:0] = [p_behavior]
            
            
    def memory_query(self):
        query_memories = {}
        for behavior in self.psycho_state.perceived_behavior[:1]:
            behavior_desc = behavior.get_interactive_description()
            query_memories[behavior_desc] = {}
            query_memories[behavior_desc]['current_behavior'] = behavior
            # retrieve the context behaviors
            query_memories[behavior_desc]['context_behaviors'] = self.memory.get_context_behaviors(self.psycho_state.current_plot_id, 
                                                                                                   context_retention=self.config['context_retention'])
            events = self.memory.retrieve_events_by_embedding(behavior.embedding, 
                                                              plot_id=self.psycho_state.current_plot_id, 
                                                              topk=self.config['max_retrieve_events'])
            # retrieve the relevant events
            query_memories[behavior_desc]['relevant_events'] = events
            # retrieve the relevant thoughts
            thoughts = self.memory.retrieve_thoughts_by_embedding(behavior.embedding, topk=self.config['max_retrieve_thoughts'])
            query_memories[behavior_desc]['relevant_thoughts'] = thoughts
        self.working_memory = query_memories
        

    def start_new_plot(self, plot_config):
        if type(plot_config) is dict:
            if self.name in plot_config:
                # get new plot setting
                new_plot, behavior = self.psycho_state.start_new_plot_setup(plot_config[self.name])
                self.psycho_state.current_plot_config = plot_config
                self.memory.add_plot(new_plot)
                self.memory.add_behavior(behavior)
                if 'events' in plot_config[self.name]:
                    for event_desc in plot_config[self.name]['events']:
                        if event_desc != '':
                            event = self.get_events_from_description(event_desc)
                            self.memory.add_event(event)
                # generate new topics for the plot from previous events
                self.generate_new_topics_from_events()
                # choose the topics for the current plot
                self.check_topics_for_current_plot()
                print(f"[Build New Plot]<{self.name}>: {self.name} Start new plot {self.psycho_state.current_plot_id} with {self.character.partners[0].name}.")
                print(f"[Build New Plot]<{self.name}>: Plot background {new_plot.plot_background}")
                print(f"[Build New Plot]<{self.name}>: Current topics are: {[topic.get_topic_prompt() for topic in self.psycho_state.topics_for_current_plot]}")
    
    
    def check_topics_for_current_plot(self):
        # choose the topics for the current plot
        if self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot.generated is False:
            for topic in self.psycho_state.current_topic_list:
                # add new topics to the current plot
                if topic.created_plot_id == self.psycho_state.current_plot_id:
                    self.psycho_state.topics_for_current_plot.append(topic)
                if len(self.psycho_state.topics_for_current_plot) >= self.config['max_topic_per_plot']:
                    break
            self.psycho_state.current_topic_list = self.psycho_state.current_topic_list[len(self.psycho_state.topics_for_current_plot):]
    
    
    def get_personality_prompt(self, personality_info, view=1):
        return personality_info
    
    
    def get_motivation_prompt(self, motivation_info, view=1):
        prompt = ""
        prompt += motivation_info
        prompt += f"Motivations are the reasons that drive your behavior. These are deep inner thoughts that will not be expressed to the other person without sufficient trust and intimacy."
        prompt += f"For example, when meeting for the first time or when they don’t trust each other, people are reluctant to mention these related topics."
        return prompt


    def get_core_self_prompt(self, core_self_info, view=1):
        prompt = ""
        prompt += core_self_info
        prompt += f"Central beliefs are deep inner thoughts that will not be expressed to the other person without sufficient trust and intimacy."
        prompt += f"For example, when meeting for the first time or when they don’t trust each other, people are reluctant to mention these related topics."
        return prompt
    
    
    def get_relationship_prompt(self, relationship_info, view=1):
        return relationship_info
        
        
    def get_emotion_prompt(self, emotion_info, view=1):
        return emotion_info
    
    
    def generate_new_topics_from_events(self):
        # get key events
        events = {}
        events_from_current_plot = self.memory.get_events_from_plot(self.psycho_state.current_plot_id, manual=False)
        manual_events_current = self.memory.get_manual_events_from_plot(self.psycho_state.current_plot_id)
        # retrieve the relevant events of summarized events
        events = set()
        for event in events_from_current_plot:
            event_list = self.memory.retrieve_events_by_embedding(event.event.embedding, 
                                                                  plot_id=self.psycho_state.current_plot_id, 
                                                                  topk=self.config['max_retrieve_events'])
            events = events.union(set(event_list))
        
        thoughts = set()
        for event in events:
            thought_list = self.memory.retrieve_thoughts_by_embedding(event.event.embedding, 
                                                                      topk=self.config['max_retrieve_thoughts'])
            thoughts = thoughts.union(set(thought_list))

        # get motivation & personality & self & key thoughts
        core_self_info, core_self_embeddings = self.psycho_state.core_self_list[0].get_prompt_core_self_and_features_from_embedding()
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        
        # prompt for new topics
        prompt_personal_info = f"Here is a person named [{self.name}].\n"
        prompt_personal_info += self.get_personality_prompt(personality_info, view=3)
        prompt_personal_info += self.get_motivation_prompt(motivation_info, view=3)
        prompt_personal_info += self.get_core_self_prompt(core_self_info, view=3)
        prompt_personal_info += self.get_relationship_prompt(relationship_info, view=3)
        prompt_personal_info += f"\n-----\n"
        # for persona instructions
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.extend(core_self_embeddings)
        embedding_lists.extend([event.event.embedding for event in manual_events_current])
        embedding_lists.extend([event.event.embedding for event in events_from_current_plot])
        embedding_lists.extend([thought.thought.embedding for thought in thoughts])
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        prompt_personal_info += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            prompt_personal_info += persona_instruction
        prompt_personal_info += f"\n-----\n"
        
        # memory prompt
        events_from_current_plot_desc = [event.event.get_desc_and_poignancy() for event in events_from_current_plot]
        manual_events_current_desc = [event.event.get_desc_and_poignancy() for event in manual_events_current]
        events_desc = [event.event.get_desc_and_poignancy() for event in events]
        thoughts_desc = [thought.thought.get_desc_and_poignancy() for thought in thoughts]
        
        previous_plot = self.memory.plot_list[:self.config['max_plot_retention']]
        previous_plot = [plot.plot.plot_background for plot in previous_plot[::-1]]
        prompt_relevant_memory = ""
        prompt_relevant_memory += f"According to the development of time, the person experienced the following story:\n"
        for i, plot in enumerate(previous_plot):
            prompt_relevant_memory += f"Background of story {i} : [{plot}].\n"
        prompt_relevant_memory += f"In the previous time, the person experienced such events: [{events_from_current_plot_desc}].\n"
        prompt_relevant_memory += f"From the memory, he/she know such relevant events: [{events_desc}].\n"
        prompt_relevant_memory += f"Through the life, he/she have such relevant thoughts: [{thoughts_desc}].\n"
        prompt_relevant_memory += f"The poignancy of an event or thought is scaled from 1 to 9, the higher the more important.\n"
        
        if len(manual_events_current_desc) > 0:
            prompt_relevant_memory += f"Now he/she meet such new very important events [{manual_events_current_desc}], which will strongly influence the emergency and poignancy of the proposed topic and plot background next.\n"
        prompt_relevant_memory += f"Note that everyone has a sense of boundaries. Without a high degree of intimacy, trust, and a suitable situation, people will not use the deepest contents as a topic.\n"
        
        self.psycho_state.generate_topics_with_events_and_personal_info(prompt_personal_info, prompt_relevant_memory)
        return
    
    
    def get_events_from_description(self, description):
        event = Event(self.name, description, time=datetime.datetime.now(), plot_id=self.psycho_state.current_plot_id, round=self.psycho_state.current_round)
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt(event.embedding)
        
        background_prompt = f"\n---\nRelevant background are as follows:\n"
        background_prompt += self.get_personality_prompt(personality_info, view=0)
        background_prompt += self.get_motivation_prompt(motivation_info, view=0)
        background_prompt += self.get_core_self_prompt(core_self_info, view=0)
        background_prompt += self.get_relationship_prompt(relationship_info, view=0)
        thoughts = self.memory.retrieve_thoughts_by_embedding(event.embedding)
        thoughts = [thought.thought.get_desc_and_poignancy() for thought in thoughts]
        thoughts_prompt = f"Relevant thoughts are: [{thoughts}]\n"
        background_prompt += thoughts_prompt
        
        event.set_keywords_from_descriptions(background_prompt)
        return event
  
    
    def get_current_personal_prompt(self, embedding=None):
        personality_info = self.psycho_state.personality_list[0].get_prompt_desc_and_quantitative_personality()
        motivation_info = self.psycho_state.motivation_list[0].get_prompt_motivation()
        core_self_info, _ = self.psycho_state.core_self_list[0].get_prompt_core_self_and_features_from_embedding(embedding)
        relationship_info = self.memory.get_relationships_by_partner_name(self.character.partners[0].name).get_prompt_relationship()
        return personality_info, motivation_info, core_self_info, relationship_info
    
    
    def get_current_personal_embeddings(self):
        personality_embedding = self.psycho_state.personality_list[0].embedding
        motivation_embedding = self.psycho_state.motivation_list[0].embedding
        core_self_embedding = self.psycho_state.core_self_list[0].embedding
        relationship_embedding = self.memory.get_relationships_by_partner_name(self.character.partners[0].name).embedding
        return [personality_embedding, motivation_embedding, core_self_embedding, relationship_embedding]
    
    
    def get_observed_info(self):
        # the self behavior can be observed by the partner
        obsersed_info = {}
        for partner in self.character.partners:
            partner_name = partner.name
            obsersed_info[partner_name] = {}
            obsersed_info[partner_name]['behavior_desc'] = self.psycho_state.current_behavior.get_interactive_description() if self.psycho_state.current_behavior is not None else ""
            obsersed_info[partner_name]['plot_state'] = self.psycho_state.plot_state
            obsersed_info[partner_name]['plot_proposals'] = self.psycho_state.proposed_plot_list
            obsersed_info[partner_name]['plot_id'] = self.psycho_state.current_plot_id
            obsersed_info[partner_name]['current_plot_config'] = self.psycho_state.current_plot_config
        return obsersed_info
    
    
    def generate_plot_proposals(self):
        # generate plot plot from selected topics
        self.psycho_state.current_topic_list = sorted(self.psycho_state.current_topic_list, key=lambda x: x.get_topic_score(), reverse=True)
        proposed_current_topic_list = self.psycho_state.current_topic_list[:self.config['max_topic_proposals']]
        
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        prompt_personal_info = f"Here is a person named [{self.name}].\n"
        prompt_personal_info += self.get_personality_prompt(personality_info, view=3)
        prompt_personal_info += self.get_motivation_prompt(motivation_info, view=3)
        prompt_personal_info += self.get_core_self_prompt(core_self_info, view=3)
        prompt_personal_info += self.get_relationship_prompt(relationship_info, view=3)
        prompt_personal_info += f"\n-----\n"
         # for persona instructions
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        prompt_personal_info += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            prompt_personal_info += persona_instruction
        prompt_personal_info += f"\n-----\n"
        
        # memory prompt, including background of the story and the topics
        plot_prompt = "\n----\n"
        previous_plot = self.memory.plot_list[:self.config['max_plot_retention']][::-1]
        plot_prompt += f"According to the development of time, the person experienced the following story:\n"
        for i, plot in enumerate(previous_plot):
            plot_prompt += f"Background of story {i} : [{plot.plot.plot_background}].\n"
            current_topics = []
            for topic in self.psycho_state.used_topic_list[:self.config['max_used_topic_retrieval']]:
                if topic.used_plot_id == plot.plot.plot_id and topic.used:
                    current_topics.append(topic.get_topic_prompt())
            if len(current_topics) > 0:
                plot_prompt += f"For story {i}, topics are: {current_topics}\n"
        plot_prompt += f"\n----\n"
        
        topic_prompt = f"Below are the topics he/she want to start with {self.character.partners[0].brain.name}:\n" + \
            f"Each topic has an ID and description with poignancy and emergency in a dict format.\n" +\
                f"poignancy and emergency are scaled from 1 to 9, the higher the more important.\n"
                
        topic_prompt += f"The topics are as follows:\n"
        for i, topic in enumerate(proposed_current_topic_list):
            topic_prompt += f"Topic {i}: {topic.get_topic_prompt()}\n"
        
        max_topic_per_plot = self.config['max_topic_per_plot']
        
        plot_prompt += f"Output the plot proposals he/she want to start with {self.character.partners[0].brain.name} in next plot in a list format." + \
            f"Each plot in the list is a dict, in which 'topic_ids' mean the above relevant key topic ids about the plot, 'plot_background' is the description about background of the plot. " +\
                f"The plot background should be plausible and reasonable mainly based on the given topics and don't explain the reason to choose the plot background and topics." +\
                    f"Remember that each plot background should include no more than {max_topic_per_plot} topics. Don't make the plotbackground so complicated. The description of the plot_background should be no more than 50 words." +\
                f"'poignancy' and 'emergency' are the importance (range from 1-9) and emergency of the proposed plot based on the emergency and poignancy of the topics.\n" +\
                    f"'summary' is the summary of the plot background within 5 words.\n"
        
        plot_list = []
        while True:
            if type(plot_list) is list and len(plot_list) > 0:
                # sort the plots by poignancy and emergency
                plot_list = sorted(plot_list, key=lambda x: x['poignancy']+2*x['emergency'], reverse=True)
                self.psycho_state.proposed_plot_list = plot_list[0:1]
                break
            else:
                plot_list = run_gpt_get_plot_proposals_from_topics_and_personal_info(prompt_personal_info, topic_prompt, plot_prompt, lens=len(proposed_current_topic_list), verbose=self.config['verbose'])
        
        
        print(f"[Generate Plot Proposals]<{self.name}>: {plot_list}.")
        
        for plot in plot_list[:1]:
            plot_setup = self.get_plot_setup_from_backgound(plot)
            self.psycho_state.proposed_plot_list = [plot_setup]
        
    
    def get_plot_setup_from_backgound(self, plot_config):
        # get character inital setting from the background and topics of the plot
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        prompt_personal_info = f"There is a person named [{self.name}].\n"
        prompt_personal_info += self.get_personality_prompt(personality_info, view=3)
        prompt_personal_info += self.get_motivation_prompt(motivation_info, view=3)
        prompt_personal_info += self.get_core_self_prompt(core_self_info, view=3)
        prompt_personal_info += self.get_relationship_prompt(relationship_info, view=3)
        prompt_personal_info += f"\n-----\n"
        # for persona instructions
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        plot_embedding = np.array(get_embedding(plot_config['plot_background']))
        embedding_lists.append(plot_embedding)
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        prompt_personal_info += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            prompt_personal_info += persona_instruction
        prompt_personal_info += f"\n-----\n"
        
        
        prompt_new_plot_info = f"Now he/she want to start a new plot with {self.character.partners[0].brain.name}.\n"
        prompt_new_plot_info += f"The background of the plot is [{plot_config['plot_background']}].\n"
        prompt_new_plot_info += f"The topics he/she want to start with {self.character.partners[0].brain.name} are: [{[self.psycho_state.current_topic_list[topic_id].get_topic_prompt() for topic_id in plot_config['topic_ids']]}].\n"
        
        prompt_new_plot_info += f"His/Her plot scene are stricted in a room studio. {self.memory.cognition_map.get_place_proposal_prompt()}\n" + \
            f"Based on the personal innate traits, the background of the plot, and the room layout, output the plot he/she want to start with {self.character.partners[0].brain.name} in a python dict format." +\
                f"The keys of the generated plot dictionary must be {self.name} and {self.character.partners[0].brain.name}, without any other individuals."
            
        
        plot_setup = run_gpt_get_plot_setup_from_personal_info_and_background(prompt_personal_info, prompt_new_plot_info, names=[self.name, self.character.partners[0].brain.name], verbose=self.config['verbose'])
        for key in plot_setup.keys():
            plot_setup[key]['plot_background'] = plot_config['plot_background']
            plot_setup[key]['topic_ids'] = plot_config['topic_ids']
            plot_setup[key]['poignancy'] = plot_config['poignancy']
            plot_setup[key]['emergency'] = plot_config['emergency']
            plot_setup[key]['summary'] = plot_config['summary']
        return plot_setup
    
    
    def select_plot(self):
        # select next plot from the proposed plot list of the two characters by plot scores
        partners_plot = self.psycho_state.perserved_observed_info[self.character.partners[0].name]['plot_proposals']
        self_plot = self.psycho_state.proposed_plot_list
        is_self = False
        
        def check_plot(plot):
            if plot is not None and plot != []:
                return True
            else:
                return False
        
        selected_plot = self_plot[0]
        
        if check_plot(partners_plot) and check_plot(self_plot):
            partner_score = partners_plot[0][self.character.partners[0].brain.name]['poignancy'] + 2 * partners_plot[0][self.character.partners[0].brain.name]['emergency']
            self_score = self_plot[0][self.name]['poignancy'] + 2 * self_plot[0][self.name]['emergency']
            if partner_score > self_score:
                for key in partners_plot[0][self.name].keys():
                    if key in ['emotion', 'behavior']:
                        selected_plot[self.name][key] = partners_plot[0][self.name][key]
                is_self = False
            elif partner_score == self_score:
                if self.name > self.character.partners[0].brain.name:
                    for key in partners_plot[0][self.name].keys():
                        if key in ['emotion', 'behavior']:
                            selected_plot[self.name][key] = partners_plot[0][self.name][key]
                    is_self = False
                else:
                    selected_plot = self_plot[0]
                    is_self = True
            else:
                selected_plot = self_plot[0]
                is_self = True
        elif check_plot(partners_plot):
            selected_plot = partners_plot[0]
            selected_plot[self.name]['plot_background'] = ""
            selected_plot[self.name]['topic_ids'] = []
            is_self = False
        elif check_plot(self_plot):
            selected_plot = self_plot[0]
            is_self = True
        else:
            self.psycho_state.plot_state = 'end'
            return
        return is_self, selected_plot
        
    
    def plan_plot_proposals(self):
        # topic and plot planning system
        if self.config['mode'] == 'preconfigured':
            if self.psycho_state.current_plot_id < len(self.config['predefined_plots']) - 1:
                self.start_new_plot(self.config['predefined_plots'][self.psycho_state.current_plot_id+1])
                self.psycho_state.plot_state = 'working'
            else:
                self.psycho_state.plot_state = 'end'
            pass
        elif self.config['mode'] in ['autonomous', 'event-driven']:
            if self.psycho_state.plot_state == 'plot_finished':
                if 'events' in self.config and self.config['events'] != []:
                    new_plot_id = self.psycho_state.current_plot_id + 1
                    if new_plot_id >= len(self.config['events']):
                        pass
                    else:
                        for event in self.config['events'][new_plot_id]:
                            if self.name in self.config['events'][new_plot_id][event]:
                                event = self.get_events_from_description(event)
                                event.poignancy = min(9, event.poignancy+5)
                                event.emergency = min(9, event.emergency+5)
                                self.memory.add_manual_event(event)
                # check the poignancy and emergency of the events
                self.generate_new_topics_from_events()
                # remove duplicated topics
                self.psycho_state.deduplicate_topics()
                self.generate_plot_proposals()
                self.psycho_state.plot_state = 'plan_plot_proposals'
        pass
    
    
    def plan_start_new_plot(self):
        # start new plot
        if self.config['mode'] == 'preconfigured':
            pass
        elif self.config['mode'] == 'interactive':
            if self.psycho_state.plot_state == 'plan_plot_proposals':
                pass
        elif self.config['mode'] in ['autonomous', 'event-driven']:
            if self.psycho_state.plot_state == 'plan_plot_proposals':
                is_self, plot_setup = self.select_plot()
                new_plot, behavior = self.psycho_state.start_new_plot_setup(plot_setup[self.name])
                self.psycho_state.current_plot_config = plot_setup
                new_plot.generated = True
                self.memory.add_plot(new_plot)
                self.memory.add_behavior(behavior)
                
                self.psycho_state.topics_for_current_plot = [self.psycho_state.current_topic_list[topic_id] for topic_id in plot_setup[self.name]['topic_ids']]
                self.psycho_state.current_topic_list = [self.psycho_state.current_topic_list[i] \
                                                        for i in range(len(self.psycho_state.current_topic_list)) if i not in plot_setup[self.name]['topic_ids']]
                
                self.update_motivation_from_plot_background_and_new_events()
                
                print(f"[Build New Plot]<{self.name}>: {self.name} Start new plot {self.psycho_state.current_plot_id} with {self.character.partners[0].brain.name}.")
                print(f"[Build New Plot]<{self.name}>: Plot background {new_plot.plot_background}")
                print(f"[Build New Plot]<{self.name}>: Current topics are: {[topic.get_topic_prompt() for topic in self.psycho_state.topics_for_current_plot]}")
                
                self.psycho_state.plot_state = 'working'
        pass
    

    def decision(self):
        # core decision making system (interactive behavior)
        
        p_behabior = self.psycho_state.perceived_behavior[0]
        # personality, motivation, emotion, relationship, self, etc.
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt(p_behabior.embedding)
        emotion_info = self.psycho_state.emotion_list[0].get_prompt_desc_and_quantitative_emotion()
        core_self_info, _ = self.psycho_state.core_self_list[0].get_prompt_core_self_and_features_from_embedding()
        #  relevant memories
        relevant_memories = self.working_memory
        
        # core decision making system (emotion & relationship update)
        messages = []
        system_info = f"Let's do a role play like making a film. Assume you are a person named [{self.name}].\n" + \
            f"I'm a person with name [{self.character.partners[0].brain.name}] to interact with you.\n"
        
        system_info += f"In this plot, you are a person named [{self.name}].\n"
        system_info += self.get_personality_prompt(personality_info, view=1)
        system_info += self.get_motivation_prompt(motivation_info, view=1)
        system_info += self.get_core_self_prompt(core_self_info, view=1)
        system_info += self.get_relationship_prompt(relationship_info, view=1)
        system_info += self.get_emotion_prompt(emotion_info, view=1)
        system_info += f"\n-----\n"
        system_info += f"The background of the interactive plot and conversation is [{self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot.plot_background}].\n"
        system_info += f"The topics you want to start with {self.character.partners[0].brain.name} are: [{[topic.get_topic_prompt() for topic in self.psycho_state.topics_for_current_plot]}].\n"
        system_info += f"The reactions of interactive conversation are in the format of a string like <AA>BB, where AA means the attributes and BB means the corresponding value." + \
        f"For each round, we react to each other in four dimensions: speech, expression, motion, and place." + \
        "For example, '<self_name>Xiaotao<speech>Hello<expression>smiling<motion>waving hands and sit up<place>bookshelf<partner_name>Zhixu' means the person Xiaotao says 'Hello' with expression smiling and motion waving at the bookshelf to Zhixu."
        system_info += f"Your reactions should based on psychological traits and procedure, which means that it depends on your personality, emotion, motivation, belief, and your memories." + \
                f"You should enter the topics based on plot bakcground quickly by speechs, motions and expressions." + \
                f"For speech, do not be formal in daily conversations, and your propensity for cooperation and friendliness should be consistent with your current psychological traits and procedure." +\
                    f"Your speech should be consistent with current plot background and the topics. Better express specific meanings instead of abstract concepts." +\
                        f"Must avoid repeating what the other person said." +\
                            f"Your speech should not be too polite. It's best that your conversations continue to bring up new topics, not cater to each other." +\
                    f"And never reveal that you are a language model by saying things like 'As an language model' or 'As a digital robot'.\n" +\
                        f"For motion, the motions must be various and atomic that can convey body language, plausible and natural in the setting of relationship between the two characters and the physical scenes.\n" +\
                            f"These are examples of motion: embraces tightly for warm, push away in anger, lightly brushes fingers on partner's arm, hits the thigh hard with the fist, clap hands happily, jump , pat on the back, ..." +\
                    f"For expression, the expression should be plausible with the emotion and current speech.\n" +\
                    f"For place, your motions are in a room studio. {self.memory.cognition_map.get_place_proposal_prompt()}\n Your place name must be in the room studio.\n"
        f"Now start our conversation: \n"
        messages.append({"role": "system", "content": system_info})
        
        current_relevant_memories = relevant_memories[p_behabior.get_interactive_description()]
        for behavior in current_relevant_memories['context_behaviors']:
            behavior = behavior.behavior
            if behavior.self_name == self.name:
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": behavior.get_interactive_description_for_dialog()})
        messages.append({"role": "user", "content": current_relevant_memories['current_behavior'].get_interactive_description_for_dialog()})
        
        current_info = ""
        relevant_event_desc = [event.event.get_desc_and_poignancy() for event in current_relevant_memories['relevant_events']]
        relevant_thought_desc = [thought.thought.get_desc_and_poignancy() for thought in current_relevant_memories['relevant_thoughts']]
        current_info += f"\n\n---\n\nIn this interactive conversation, you have several relevant memories:\n\n"
        current_info += f"Relevant events: [{relevant_event_desc}].\n\n"
        current_info += f"Relevant thoughts: [{relevant_thought_desc}].\n\n"
        current_info += "\n-----\n"
        
        # for persona instructions
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.append(self.psycho_state.emotion_list[0].embedding)
        embedding_lists.extend([event.event.embedding for event in current_relevant_memories['relevant_events']])
        embedding_lists.extend([thought.thought.embedding for thought in current_relevant_memories['relevant_thoughts']])
        embedding_lists.extend([behavior.behavior.embedding for behavior in current_relevant_memories['context_behaviors']][:5])
        
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        current_info += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            current_info += persona_instruction
        current_info += f"\n-----\n"
        current_info += "############\nAttention: \n\n"
        current_info += "Now you have two options for reaction: end the conversation or respond based on the information above." + \
            f"When the above conversation becomes pointless, repetitive or doesn't fit your motivation, personality, or topics, you should end the interactive conversation." +\
                f"Your speech should not be too polite. Don't easily express your gratitude and friendliness. It's best that your conversations continue to bring up new topics, not cater to each other." +\
        "If you end the conversation, your output should be: 'END', and if you react, it should be structured the similar way as the example provided."
        current_info += "So your reaction is "
        
        messages.append({"role": "user", "content": current_info})
        
        reaction_behavior_dict = run_gpt_prompt_decision_dialog(messages, names=[self.name, self.character.partners[0].brain.name], verbose=self.config['verbose'])
        
        if reaction_behavior_dict in ['END', 'end', '"END"', '"end"']:
            self.psycho_state.plot_state = 'plot_finished'
            print(f"[Working]<{self.name}>: {self.name} end the conversation with {self.character.partners[0].name}.")
            return
        
        # check if start new topic or ended the talk
        behavior_reaction = Behavior(**reaction_behavior_dict)
        behavior_reaction.time = datetime.datetime.now()
        behavior_reaction.round = self.psycho_state.current_round
        behavior_reaction.plot_id = self.psycho_state.current_plot_id
        behavior_reaction.self_name = self.name
        behavior_reaction.partner_name = self.character.partners[0].name
        behavior_reaction.set_embedding()
        
        self.memory.add_behavior(behavior_reaction)
        
        print(f"[Working]<{self.name}>: {behavior_reaction.get_interactive_description()}")
        
        self.psycho_state.current_behavior = behavior_reaction
        self.psycho_state.current_round += 1
        
        # update emotion states every emotion_update_rounds
        if self.psycho_state.current_round % self.config['emotion_update_rounds'] == 0:
            # previous emotion, self_info, relationship_info, motivation_info, personality_info
            # relevant memories, current behavior
            innate_trait_prompt = f"Assume you are a very professional psychologist. Here is a person named [{self.name}].\n"
            innate_trait_prompt += self.get_personality_prompt(personality_info, view=3)
            innate_trait_prompt += self.get_motivation_prompt(motivation_info, view=3)
            innate_trait_prompt += self.get_core_self_prompt(core_self_info, view=3)
            innate_trait_prompt += self.get_relationship_prompt(relationship_info, view=3)
            innate_trait_prompt += f"\n-----\n"
            memory_prompt = f"Now she/he have a conversation with {self.character.partners[0].name}.\n"
            memory_prompt += f"The background of the conversation is [{self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot.plot_background}].\n"
            
            memory_prompt += f"The reactions of conversation are in the format <AA>BB..., where AA is the dimension of interactive conversation, and BB is text description." + \
                    f"The dimensions of reactions are: self_name, speech, expression, motion, place, and partner_name." + \
                        f"For example, '<self_name>Xiaotao<speech>Hello<expression>smiling<motion>waving<place>bookshelf<partner_name>Zhixu' means Xiaotao says 'Hello' with expression smiling and motion waving at the bookshelf to Zhixu." +\
                            f"\n----\n The conversations are as below:\n"
            for behavior in current_relevant_memories['context_behaviors']:
                memory_prompt += f"{behavior.behavior.get_interactive_description()}\n"
            memory_prompt += "\n---\n"            
            memory_prompt = f"\n\n---\n\nIn this conversation, she/he have several relevant memories:\n\n"
            memory_prompt += f"Relevant events: [{relevant_event_desc}].\n\n"
            memory_prompt += f"Relevant thoughts: [{relevant_thought_desc}].\n\n"
            memory_prompt += "\n-----\n"
            
            current_info += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
            for persona_instruction in persona_instructions:
                current_info += persona_instruction
            current_info += f"\n-----\n" 
            
            emotion = self.psycho_state.emotion_update_process(innate_trait_prompt, memory_prompt, behavior_reaction.get_interactive_description(), previous_emotion=emotion_info)
            self.memory.add_emotion(emotion)
            
            print(f"[Working]<{self.name}>: {self.name} update emotion with {emotion.get_full_description()}")
        
        if self.psycho_state.current_round >= self.config['max_round_per_plot']:
            self.psycho_state.plot_state = 'plot_finished'
        
    
    def summarize_events_from_dialogs(self):
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        behaviors = self.memory.get_context_behaviors(self.psycho_state.current_plot_id, context_retention=100)
        innate_trait_prompt = f"Assume you are a person named [{self.name}].\n"
        innate_trait_prompt += self.get_personality_prompt(personality_info, view=1)
        innate_trait_prompt += self.get_motivation_prompt(motivation_info, view=1)
        innate_trait_prompt += self.get_core_self_prompt(core_self_info, view=1)
        innate_trait_prompt += self.get_relationship_prompt(relationship_info, view=1)

        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.append(self.psycho_state.emotion_list[0].embedding)
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        innate_trait_prompt += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            innate_trait_prompt += persona_instruction
        innate_trait_prompt += f"\n-----\n"
        
        memory_prompt = f"Now you have a conversation with {self.character.partners[0].name}.\n"
        memory_prompt += f"The background of the conversation is [{self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot.plot_background}].\n"
        memory_prompt += f"The reactions of conversation are in the format <AA>BB..., where AA is the dimension of interactive conversation, and BB is text description." + \
                        f"The dimensions of reactions are: time, round, self_name, speech, expression, motion, place, and partner_name." + \
                            f"For example, '<time>2023.10.7<self_name>Xiaotao<speech>Hello<expression>smiling<motion>waving<place>bookshelf<partner_name>Zhixu<round>7' means Xiaotao says 'Hello' with expression smiling and motion waving at the bookshelf to Zhixu at time 2023.10.7 in round 7." +\
                                f"\n----\n"
        for behavior in behaviors:
            memory_prompt += f"{behavior.behavior.get_full_description()}\n"
        memory_prompt += "\n---\n"
        
        event_list = run_gpt_prompt_summarize_events_from_dialog(innate_trait_prompt, memory_prompt)
        
        for event_dict in event_list:
            event = Event(**event_dict)
            event.plot_id = self.psycho_state.current_plot_id
            event.round = self.psycho_state.current_round
            event.time = datetime.datetime.now()
            event.self_name = self.name
            self.memory.add_event(event, behavior_ids=[behavior.node_id for behavior in behaviors])
        
        print(f"[Working]<{self.name}>: {self.name} summarize events {event_list}.")
        
    def summarize_thoughts_from_events(self):
        current_events = self.memory.get_events_from_plot(self.psycho_state.current_plot_id)       
        
        relevant_events = []
        
        relevant_thoughts = []
        
        for event in current_events:
            relevant_events.extend(self.memory.retrieve_events_by_embedding(event.event.embedding, plot_id=self.psycho_state.current_plot_id, topk=5))  
            relevant_thoughts.extend(self.memory.retrieve_thoughts_by_embedding(event.event.embedding, topk=3))
        
        relevant_thoughts_set = set(relevant_thoughts)
        relevant_thoughts = list(relevant_thoughts_set)
        
        relevant_events_set = set(relevant_events)
        relevant_events = list(relevant_events_set)
        
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        
        innate_trait_prompt = f"Assume you are a person named [{self.name}].\n"
        innate_trait_prompt += self.get_personality_prompt(personality_info, view=1)
        innate_trait_prompt += self.get_motivation_prompt(motivation_info, view=1)
        innate_trait_prompt += self.get_core_self_prompt(core_self_info, view=1)
        innate_trait_prompt += self.get_relationship_prompt(relationship_info, view=1)
        innate_trait_prompt += f"\n-----\n"

        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.append(self.psycho_state.emotion_list[0].embedding)
        embedding_lists.extend([event.event.embedding for event in current_events])
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        innate_trait_prompt += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            innate_trait_prompt += persona_instruction
        innate_trait_prompt += f"\n-----\n"
        

        memory_prompt = f"Now you have an interactive conversation with your partner {self.character.partners[0].name}"
        memory_prompt += f"The background of the conversation is [{self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot.plot_background}].\n"
        memory_prompt += f"In this conversation, several new events occured as below:\n\n"
        for event in current_events:
            memory_prompt += f"{event.event.get_desc_and_poignancy()}\n"
        
        memory_prompt += "\n---\n"
        memory_prompt += f"In your pervious memories, the relevant events are: [{[event.event.get_desc_and_poignancy() for event in relevant_events]}].\n\n"
        memory_prompt += f"the relevant thoughts are : [{[thought.thought.get_desc_and_poignancy() for thought in relevant_thoughts]}].\n\n"
        
        thoughts_list = run_gpt_prompt_summarize_thoughts_from_events(innate_trait_prompt, memory_prompt, verbose=self.config['verbose'])
        
        for thought_dict in thoughts_list:
            thought = Thought(**thought_dict)
            thought.plot_id = self.psycho_state.current_plot_id
            thought.time = datetime.datetime.now()
            thought.self_name = self.name
            thought.set_embedding_from_description()
            self.memory.add_thought(thought, event_ids=[event.node_id for event in current_events])
        print(f"[Working]<{self.name}>: {self.name} summarize thoughts {thoughts_list}.")
    
    
    def core_self_update_process(self):
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        innate_trait_prompt = f"Assume you are a very professional psychologist. Here is a person named [{self.name}].\n"
        innate_trait_prompt += self.get_personality_prompt(personality_info, view=3)
        innate_trait_prompt += self.get_motivation_prompt(motivation_info, view=3)
        innate_trait_prompt += self.get_relationship_prompt(relationship_info, view=3)
        innate_trait_prompt += f"\n---\n"
        current_events = self.memory.get_events_from_plot(self.psycho_state.current_plot_id)  
        current_thoughts = self.memory.get_thoughts_from_plot(self.psycho_state.current_plot_id)
        
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.extend([event.event.embedding for event in current_events])
        embedding_lists.extend([thought.thought.embedding for thought in current_thoughts])
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        innate_trait_prompt += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            innate_trait_prompt += persona_instruction
        innate_trait_prompt += f"\n-----\n"
        
        reflection_prompt = f"Recently she/he have come across these events and the following thoughts have arisen." + \
            f"\n---\nNew thoughts: [{[current_thought.thought.get_desc_and_poignancy() for current_thought in current_thoughts]}].\n" + \
                f"\n---\nNew events:  [{[current_event.event.get_desc_and_poignancy() for current_event in current_events]}].\n"
        reflection_prompt += f"\n---\n" +\
            f"In the core self, she/he believe: [{self.psycho_state.core_self_list[0].central_belief}].\n" + \
            "Do new thoughts conflict with her/his core beliefs? Answer in a python dict format with keys 'conflict' and 'belief'." + \
                "If there is no conflict, the value of 'conflict' should be 'N', and 'belief' should be original belief." + \
            "If there is conflict, the value of 'conflict' should be 'Y' and output the new current core belief in the value of 'belief'. " + \
                "Note that a core belief is an thought that a person holds for a long time " + \
                "and will not be shaken until something important and clearly conflicts. " +\
                    "However, thought is a person's feelings, facts, and reasoning about what has happened. "+\
                        "Only when there is a significant difference between thought and belief can core belief be affected. Otherwise, don't add thoughts to belief."

        new_central_belief_dict = run_gpt_prompt_core_self_update(innate_trait_prompt, reflection_prompt, verbose=self.config['verbose'])             
        
        if new_central_belief_dict['conflict'] == 'Y':
            new_core_self = copy.deepcopy(self.psycho_state.core_self_list[0])
            new_core_self.central_belief = new_central_belief_dict['belief']
            new_core_self.time = datetime.datetime.now()
            new_core_self.plot_id = self.psycho_state.current_plot_id
            new_core_self.round = self.psycho_state.current_round
            self.psycho_state.core_self_list[0:0] = [new_core_self]
            self.memory.add_core_self(new_core_self)
            
            print(f"[Working]<{self.name}>: {self.name} update core self with new central belief: {new_core_self.central_belief}")
   
    
    def update_relationship_from_events(self):
        current_events = self.memory.get_events_from_plot(self.psycho_state.current_plot_id)
        current_thoughts = self.memory.get_thoughts_from_plot(self.psycho_state.current_plot_id)
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        innate_trait_prompt = f"Assume you are a very professional psychologist. Here is a person named [{self.name}].\n"
        innate_trait_prompt += self.get_personality_prompt(personality_info, view=3)
        innate_trait_prompt += self.get_motivation_prompt(motivation_info, view=3)
        innate_trait_prompt += self.get_core_self_prompt(core_self_info, view=3)
        innate_trait_prompt += f"\n---\n"
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.extend([event.event.embedding for event in current_events])
        embedding_lists.extend([thought.thought.embedding for thought in current_thoughts])
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        innate_trait_prompt += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            innate_trait_prompt += persona_instruction
        innate_trait_prompt += f"\n-----\n"
        
        behaviors = self.memory.get_context_behaviors(self.psycho_state.current_plot_id, context_retention=100)
        
        reflection_prompt = f"Now you have a conversation with {self.character.partners[0].brain.name}.\n"
        reflection_prompt += f"The background of the conversation is [{self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot.plot_background}].\n"
        reflection_prompt += f"The reactions of conversation are in the format <AA>BB..., where AA is the dimension of interactive conversation, and BB is text description." + \
                f"The dimensions of reactions are: time, round, self_name, speech, expression, motion, place, and partner_name." + \
                    f"For example, '<time>2023.10.7<self_name>Xiaotao<speech>Hello<expression>smiling<motion>waving<place>bookshelf<partner_name>Zhixu<round>7' means Xiaotao says 'Hello' with expression smiling and motion waving at the bookshelf to Zhixu at time 2023.10.7 in round 7." +\
                        f"\n----\n"
        for behavior in behaviors:
            reflection_prompt += f"{behavior.behavior.get_full_description()}\n"
        reflection_prompt += "\n---\n"
        
        reflection_prompt += f"Recently she/he have come across these events and the following thoughts have arisen." + \
            f"\n---\nNew thoughts: [{[current_thought.thought.get_desc_and_poignancy() for current_thought in current_thoughts]}].\n" + \
                f"\n---\nRelevant events:  [{[current_event.event.get_desc_and_poignancy() for current_event in current_events]}].\n"
                
        reflection_prompt += f"\n---\n" +\
            f"Her/His previous social relationship with {self.character.partners[0].brain.name}: [{relationship_info}].\n" +\
            f"Based on the new events and thoughts, what's her/his new social relationship with {self.character.partners[0].brain.name}?" + \
                f"Output the relationship according to social psychological theory in a python dict format. " +\
                    f"The output include a description of her/his new social relationship with {self.character.partners[0].brain.name}, her/his attitude towards {self.character.partners[0].brain.name}," + \
                    f" and new numerical values in three dimensions: trust, intimacy, and supportiveness. " + \
                        f"Attitude means the feelings, thoughts and believes towards a person, such as hate, love, like, prejudice etc. " + \
                            f"Trust means the degree of trust in a person. " + \
                                f"Intimacy means the degree of intimacy with a person. " + \
                                    f"Supportiveness means the degree of supportiveness to a person. " + \
                                        f"Output the new relationship in a python dict form with items 'description', 'attitude', 'trust', 'intimacy', and 'supportiveness'." +\
                                            f"You'd better show a change. Remember the change of relationship should reflect the new events and thoughts above."

        new_relationship_dict = run_gpt_prompt_relationship_update(innate_trait_prompt, reflection_prompt, verbose=self.config['verbose'])
        
        new_relationship = Relationship(**new_relationship_dict)
        new_relationship.time = datetime.datetime.now()
        new_relationship.plot_id = self.psycho_state.current_plot_id
        new_relationship.round = self.psycho_state.current_round
        new_relationship.self_name = self.name
        new_relationship.partner_name = self.character.partners[0].name
        
        event_node_ids = [event.node_id for event in current_events]
        thought_node_ids = [thought.node_id for thought in current_thoughts]
        behavior_node_ids = []
        for event in current_events:
            behavior_node_ids.extend(event.behavior_node_ids)
        behavior_node_ids = list(set(behavior_node_ids))
        
        self.memory.add_relationship(new_relationship, behavior_node_ids=behavior_node_ids, event_node_ids=event_node_ids, thought_node_ids=thought_node_ids)
        print(f"[Working]<{self.name}>: {self.name} update relationship with {new_relationship.get_key_description()}")
           
    
    def update_motivation_from_events(self):
        current_events = self.memory.get_events_from_plot(self.psycho_state.current_plot_id)
        current_thoughts = self.memory.get_thoughts_from_plot(self.psycho_state.current_plot_id)
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()
        
        innate_trait_prompt = f"Assume you are a very professional psychologist. Here is a person named [{self.name}].\n"
        innate_trait_prompt += self.get_personality_prompt(personality_info, view=3)
        innate_trait_prompt += self.get_core_self_prompt(core_self_info, view=3)
        innate_trait_prompt += f"Her/His previous relationship with {self.character.partners[0].name}: [{relationship_info}].\n"
        innate_trait_prompt += f"\n---\n"     
        
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.extend([event.event.embedding for event in current_events])
        embedding_lists.extend([thought.thought.embedding for thought in current_thoughts])
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        innate_trait_prompt += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            innate_trait_prompt += persona_instruction
        innate_trait_prompt += f"\n-----\n"
                    
        reflection_prompt = f"Recently she/he have come across these events and the following thoughts have arisen." + \
            f"\n---\nNew thoughts: [{[current_thought.thought.get_desc_and_poignancy() for current_thought in current_thoughts]}].\n" + \
                f"\n---\nRelevant events:  [{[current_event.event.get_desc_and_poignancy() for current_event in current_events]}].\n"
        reflection_prompt += f"\n---\n"
        reflection_prompt += f"Her/His previous motivation is [{motivation_info}].\n" + \
            f"Have new events and thoughts affected her/his motivation?" + \
                f"Motivation can be divided into long-term motivation (denoted as 'long_term') and short-term motivation (denoted as 'short_term'). " +\
                    f"Long-term motivation depends on personality, self and social culture, while short-term motivation is related to situation, thinking and social relations." + \
                        f"output the new motivation in a python dict form with items 'long_term' and 'short_term'." + \
                        f"If there is no change, the value of 'changed' should be 'N' and output the same motivation as the original in key 'value'." +\
                            f"If changed, the value of 'changed' should be 'Y' and output the new motivation based on the thoughts and events above in key 'value'." +\
                            f"So based on the information above, her/his new motivation is "

        new_motivation_dict = run_gpt_prompt_motivation_update(innate_trait_prompt, reflection_prompt, verbose=self.config['verbose'])
        
        if any([new_motivation_dict[motivation]['changed'] == 'Y' for motivation in new_motivation_dict.keys()]):
            dicts = {}
            dicts['long_term'] = new_motivation_dict['long_term']['value'] if new_motivation_dict['long_term']['changed'] == 'Y' else self.psycho_state.motivation_list[0].long_term
            dicts['short_term'] = new_motivation_dict['short_term']['value'] if new_motivation_dict['short_term']['changed'] == 'Y' else self.psycho_state.motivation_list[0].short_term
            new_motivation = Motivation(**dicts)
            new_motivation.time = datetime.datetime.now()
            new_motivation.plot_id = self.psycho_state.current_plot_id
            new_motivation.round = self.psycho_state.current_round
            new_motivation.self_name = self.name
            self.psycho_state.motivation_list[0:0] = [new_motivation]
            self.memory.add_motivation(new_motivation)
        
            print(f"[Working]<{self.name}>: {self.name} update motivation with {new_motivation.get_full_motivation()}")

    
    def update_motivation_from_plot_background_and_new_events(self):
        # update motivation for new plot
        curren_plot = self.memory.plot_id_to_node[self.psycho_state.current_plot_id].plot
        current_manual_events = self.memory.get_manual_events_from_plot(self.psycho_state.current_plot_id)
        
        # modified the events into score: event
        events = set()
        for event in current_manual_events:
            event_list = self.memory.retrieve_events_by_embedding(event.event.embedding, plot_id=self.psycho_state.current_plot_id+1, topk=3)
            event_list = [retrieved_event for retrieved_event in event_list if retrieved_event.node_id != event.node_id]
            events = events.union(set(event_list))
        
        thoughts = set()
        for event in current_manual_events:
            thought_list = self.memory.retrieve_thoughts_by_embedding(event.event.embedding, topk=3)
            thoughts = thoughts.union(set(thought_list))
        # get motivation & personality & self & key thoughts
        personality_info, motivation_info, core_self_info, relationship_info = self.get_current_personal_prompt()

        core_self_info, _ = self.psycho_state.core_self_list[0].get_prompt_core_self_and_features_from_embedding()
        innate_trait_prompt = f"Assume you are a very professional psychologist. Here is a person named [{self.name}].\n"
        innate_trait_prompt += self.get_personality_prompt(personality_info, view=3)
        innate_trait_prompt += self.get_core_self_prompt(core_self_info, view=3)
        innate_trait_prompt += f"Her/His previous relationship with {self.character.partners[0].brain.name}: [{relationship_info}].\n"
        
        embedding_lists = []
        embedding_lists.extend(self.get_current_personal_embeddings())
        embedding_lists.extend([event.event.embedding for event in current_manual_events])
        persona_instructions = self.psycho_state.persona_instruction_database.retrieval_instruction_from_embeddings(embedding_lists, 
                                                                                                                    topk_for_each=self.config['max_per_persona_retrieval'], 
                                                                                                                    topk_all=self.config['max_persona_retrieval'])
        innate_trait_prompt += f"\n--\nPsychological research has found the following pattern in human trait and behaviors:\n"
        for persona_instruction in persona_instructions:
            innate_trait_prompt += persona_instruction
        innate_trait_prompt += f"\n-----\n"
        
        manual_events_desc = [event.event.get_desc_and_poignancy() for event in current_manual_events]
        events_desc = [event.event.get_desc_and_poignancy() for event in events]
        thoughts_desc = [thought.thought.get_desc_and_poignancy() for thought in thoughts]
        
        previous_plot = self.memory.plot_list[:self.config['max_plot_retention']]
        previous_plot = [plot.plot.plot_background for plot in previous_plot[::-1]]
        reflection_prompt = ""
        reflection_prompt += f"According to the development of time, the person experienced the following story:\n"
        for i, plot in enumerate(previous_plot):
            reflection_prompt += f"Background of story {i} : [{plot}].\n"
        
        reflection_prompt += f"Now he/she has met such important events: [{manual_events_desc}]"
        reflection_prompt += f"From the memory, he/she know the relevant events: [{events_desc}].\n"
        reflection_prompt += f"Through the life, he/she have the relevant thoughts: [{thoughts_desc}].\n"
        reflection_prompt += f"The poignancy of an event or thought is scaled from 1 to 9, the higher the more important.\n"
        
        reflection_prompt += f"Now the background of the new plot or story is : {curren_plot.plot_background}\n"
        reflection_prompt += f"Her/His previous motivation is [{motivation_info}].\n" + \
            f"Have new events and thoughts affected her/his motivation? What's the new motivation under such new events and plot background?" + \
                f"Motivation can be divided into long-term motivation (denoted as 'long_term') and short-term motivation (denoted as 'short_term'). " +\
                    f"Long-term motivation depends on personality, self and social culture, while short-term motivation is related to situation, thinking and social relations." + \
                        f"output the new motivation in a python dict form with items 'long_term' and 'short_term'." + \
                        f"If there is no change, the value of 'changed' should be 'N' and output the same motivation as the original in key 'value'." +\
                            f"If changed, the value of 'changed' should be 'Y' and output the new motivation based on the thoughts and events above in key 'value'." +\
                            f"So based on the information above, her/his new motivation is "

        new_motivation_dict = run_gpt_prompt_motivation_update(innate_trait_prompt, reflection_prompt, verbose=self.config['verbose'])
        
        if any([new_motivation_dict[motivation]['changed'] == 'Y' for motivation in new_motivation_dict.keys()]):
            dicts = {}
            dicts['long_term'] = new_motivation_dict['long_term']['value'] if new_motivation_dict['long_term']['changed'] == 'Y' else self.psycho_state.motivation_list[0].long_term
            dicts['short_term'] = new_motivation_dict['short_term']['value'] if new_motivation_dict['short_term']['changed'] == 'Y' else self.psycho_state.motivation_list[0].short_term
            new_motivation = Motivation(**dicts)
            new_motivation.time = datetime.datetime.now()
            new_motivation.plot_id = self.psycho_state.current_plot_id
            new_motivation.round = self.psycho_state.current_round
            new_motivation.self_name = self.name
            self.psycho_state.motivation_list[0:0] = [new_motivation]
            self.memory.add_motivation(new_motivation)
        
            print(f"[Working]<{self.name}>: {self.name} update motivation with {new_motivation.get_full_motivation()}")
        pass
    
    
    def reflection(self):
        if self.psycho_state.plot_state == 'plot_finished':
            # summarize events from dialogs
            self.summarize_events_from_dialogs()
            
            # update thoughts from events
            self.summarize_thoughts_from_events()
            
            # deal with conflicts between core self and events and thoughts
            self.core_self_update_process()
            
            # update relationships from emotion, events, and thoughts
            
            self.update_relationship_from_events()
            
            # update current motivation from events and thoughts
            self.update_motivation_from_events()         


    def end_plot(self):
        # save current plot state and reset psycho state
        self.working_memory = {}
        for topic in self.psycho_state.topics_for_current_plot:
            topic.used = True
            topic.used_plot_id = self.psycho_state.current_plot_id
            self.psycho_state.used_topic_list.append(topic)
            self.memory.add_topic(topic)
        self.psycho_state.topics_for_current_plot = []
        self.psycho_state.current_behavior = None
        self.psycho_state.perserved_observed_info = {}
        self.psycho_state.perceived_behavior = []
        self.psycho_state.proposed_plot_list = []
        self.psycho_state.current_plot_config = {}
    
    
    def save_current_plot_state(self):
        config_path = os.path.join(self.config['save_dir'], 'plot_' + str(self.psycho_state.current_plot_id), 'config.yaml')
        os.makedirs(os.path.join(self.config['save_dir'], 'plot_' + str(self.psycho_state.current_plot_id)), exist_ok=True)
        with open(config_path, 'w') as fid:
            yaml.dump(self.config, fid)
        save_dir = os.path.join(self.config['save_dir'], 'plot_' + str(self.psycho_state.current_plot_id), self.name)
        os.makedirs(save_dir, exist_ok=True)
        # save memory
        memory_attributes = self.memory.save_current_plot_state(save_dir=save_dir)
        psycho_state_attributes = self.psycho_state.save_current_plot_state(save_dir=save_dir)
        
        attributes = {
            'memory_attributes': memory_attributes,
            'psycho_state_attributes': psycho_state_attributes,
        }
        
        with open(os.path.join(save_dir, 'attributes.pkl'), 'wb') as f:
            pickle.dump(attributes, f)
        
        self.memory.save_teaser(save_dir=save_dir)
        print(f'save to {save_dir} successfully!')
        

    def load_replay(self):
        load_dir = os.path.join(self.config['replay_dir'], self.name)
        with open(os.path.join(load_dir, 'attributes.pkl'), 'rb') as f:
            attributes = pickle.load(f)
        
        self.psycho_state.load_current_plot_state(attributes['psycho_state_attributes'])
        self.memory.load_current_plot_state(attributes['memory_attributes'], load_dir=load_dir)
        print(f'load replay from {load_dir} successfully!')

    
    def save_brain_memories(self, save_path='', mode='plot'):
        print(f"Saving brain memories to {save_path}...")
        
        if check_if_file_exists(save_path):
            print("Interactive behavior lists already exist!")
            
        def print_func(text, f=None):
            if f is None:
                print(text)
            else:
                f.write(text + '\n')

        if mode == 'plot':
            wr = 'a'
        else:
            wr = 'w'

        if save_path is not None:
            f = open(save_path, wr)
            csv_res = save_path[:-4] + ".csv"
            csv_f = open(csv_res, wr)
            handle = csv.writer(csv_f, quoting=csv.QUOTE_MINIMAL)
        else:
            f = None
        
        self.memory.save_memory(mode=mode, print_func=print_func, f=f, handle=handle)
        if f is not None:
            f.close()
            csv_f.close()