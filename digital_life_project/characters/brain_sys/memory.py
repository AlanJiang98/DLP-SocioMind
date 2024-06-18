"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: memory.py
Description: Memory module of the SocioMind
"""
import datetime
from typing import Any
import numpy as np
import os
import pickle
from digital_life_project.characters.brain_sys.psycho_state.behavior import Behavior
from digital_life_project.characters.brain_sys.memory_modules.social_memory import Relationship
from digital_life_project.characters.brain_sys.psycho_state.plot import Plot, Topic
from digital_life_project.characters.brain_sys.memory_modules.episodic_semantic_memory import Event, Thought
from digital_life_project.characters.brain_sys.memory_modules.cognition_map import CognitionMap
from digital_life_project.characters.brain_sys.psycho_state.emotion import Emotion
from digital_life_project.characters.brain_sys.psycho_state.core_self import Coreself
from digital_life_project.characters.brain_sys.psycho_state.motivation import Motivation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



class BehaviorNode:
  def __init__(self, node_id, node_count, type_count, created, expiration, behavior: Behavior,
               last_self_node_id=-1, last_partner_node_id=-1, next_partner_node_id=-1, next_self_node_id=-1):
    self.node_id = node_id
    self.node_count = node_count
    self.type_count = type_count
    self.created = created
    self.expiration = expiration
    self.last_accessed = self.created
    self.behavior = behavior
    self.plot_id = behavior.plot_id
    self.last_self_node_id = last_self_node_id
    self.last_partner_node_id = last_partner_node_id
    self.next_partner_node_id = next_partner_node_id
    self.next_self_node_id = next_self_node_id
        

class EventNode:
    def __init__(self,
                    node_id, node_count, type_count, created, expiration,
                    event: Event, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.expiration = expiration
        self.plot_id = plot_id
        self.last_accessed = self.created
        self.event = event
        self.behavior_node_ids = []


class RelationshipNode:
  def __init__(self,
               node_id, node_count, type_count, created, expiration,
               relationship: Relationship, plot_id):
    self.node_id = node_id
    self.node_count = node_count
    self.type_count = type_count
    self.created = created
    self.relationship = relationship
    self.last_accessed = self.created
    self.expiration = expiration
    self.plot_id = plot_id
    self.behavior_node_ids = []
    self.event_node_ids = []
    self.thought_node_ids = []


class EmotionNode:
    def __init__(self, node_id, node_count, type_count, created, expiration, emotion: Emotion, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.emotion = emotion
        self.last_accessed = self.created
        self.expiration = expiration
        self.plot_id = plot_id 


class CoreselfNode:
    def __init__(self, node_id, node_count, type_count, created, expiration, coreself: Coreself, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.coreself = coreself
        self.last_accessed = self.created
        self.expiration = expiration
        self.plot_id = plot_id


class MotivationNode:
    def __init__(self, node_id, node_count, type_count, created, expiration, motivation: Motivation, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.motivation = motivation
        self.last_accessed = self.created
        self.expiration = expiration
        self.plot_id = plot_id


class PlotNode:
    def __init__(self, node_id, node_count, type_count, created, expiration, plot: Plot, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.plot = plot
        self.last_accessed = self.created
        self.expiration = expiration
        self.plot_id = plot_id
        self.event_node_ids = []
        self.relationship_node_ids = []
        self.behavior_node_ids = []
        self.topic_node_ids = []
        self.emotion_node_ids = []
        self.core_self_node_ids = []
        self.motivation_node_ids = []
        self.thought_node_ids = []


class TopicNode:
    def __init__(self, node_id, node_count, type_count, created, expiration, topic: Topic, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.topic = topic
        self.last_accessed = self.created
        self.expiration = expiration
        self.plot_id = plot_id


class ThoughtNode:
    def __init__(self, node_id, node_count, type_count, created, expiration, thought: Thought, plot_id):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.created = created
        self.thought = thought
        self.last_accessed = self.created
        self.expiration = expiration
        self.plot_id = plot_id
        self.event_node_ids = []


class Memory:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        
        self.cognition_map = CognitionMap(config)
        
        # event, thoughts, behavior, plot, relationship
        self.id_to_node = {}
        self.plot_id_to_node = dict()    
        self.event_list = []   
        self.manual_event_list = []       
        self.behavior_list = []      
        self.thought_list = []        
        self.plot_list = []        
        self.relationship_dicts = {}       
        self.emotion_list = []      
        self.core_self_list = []      
        self.motivation_list = []       
        self.topic_list = []       
        self.current_plot_id = -1
        
        self.init_relationships(config['characters_info'][self.name]['relationships'])
        
        
    def save_current_plot_state(self, save_dir=''):
        node_path = os.path.join(save_dir, 'id_to_nodes.pkl')
        with open(node_path, 'wb') as file:
            pickle.dump(self.id_to_node, file)
        
        saved_attrs = {}
        for name in ['event_list', 'manual_event_list', 'behavior_list', 'thought_list', 'plot_list', 'emotion_list', 'core_self_list', 'motivation_list', 'topic_list']:
            saved_attrs[name] = []
            for item in getattr(self, name):
                saved_attrs[name].append(item.node_id)
        saved_attrs['current_plot_id'] = self.current_plot_id
        saved_attrs['relationship_dicts'] = {}
        for key in self.relationship_dicts.keys():
            saved_attrs['relationship_dicts'][key] = []
            for item in self.relationship_dicts[key]:
                saved_attrs['relationship_dicts'][key].append(item.node_id)
        saved_attrs['plot_id_to_node'] = {}
        for key in self.plot_id_to_node.keys():
            saved_attrs['plot_id_to_node'][key] = self.plot_id_to_node[key].node_id
        return saved_attrs
    
    
    def load_current_plot_state(self, saved_attrs, load_dir=''):
        node_path = os.path.join(load_dir, 'id_to_nodes.pkl')
        with open(node_path, 'rb') as file:
            self.id_to_node = pickle.load(file)
        
        for name in ['event_list', 'manual_event_list', 'behavior_list', 'thought_list', 'plot_list', 'emotion_list', 'core_self_list', 'motivation_list', 'topic_list']:
            setattr(self, name, [])
            for node_id in saved_attrs[name]:
                getattr(self, name).append(self.id_to_node[node_id])
        
        self.current_plot_id = saved_attrs['current_plot_id']
        self.relationship_dicts = {}
        for key in saved_attrs['relationship_dicts'].keys():
            self.relationship_dicts[key] = []
            for node_id in saved_attrs['relationship_dicts'][key]:
                self.relationship_dicts[key].append(self.id_to_node[node_id])
        
        self.plot_id_to_node = {}
        for key in saved_attrs['plot_id_to_node'].keys():
            self.plot_id_to_node[key] = self.id_to_node[saved_attrs['plot_id_to_node'][key]]
        return
        
    
    def init_relationships(self, dicts):
        for key in dicts.keys():
            relationship = Relationship.parse_relationship_from_dicts(dicts[key])
            relationship.self_name = self.name
            relationship.partner_name = key
            relationship.time = datetime.datetime.now()
            relationship.plot_id = self.current_plot_id
            self.add_relationship(relationship)
    
    def add_plot(self, plot: Plot):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.plot_list) + 1
        node_id = node_count
        created = plot.time
        plot_id = plot.plot_id
        
        plot_node = PlotNode(node_id, node_count, type_count, created, None, plot, plot_id)
        
        self.plot_list[0:0] = [plot_node]
        self.id_to_node[node_id] = plot_node
        self.plot_id_to_node[plot_id] = plot_node
        self.current_plot_id = plot_id
        return plot_node
    
    def add_event(self, event, behavior_ids=[]):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.event_list) + 1
        node_id = node_count
        created = event.time
        event_node = EventNode(node_id, node_count, type_count, created, None, event, plot_id=event.plot_id)
        self.event_list[0:0] = [event_node]
        self.id_to_node[node_id] = event_node
        plot = self.plot_id_to_node[event.plot_id]
        plot.event_node_ids[0:0] = [node_id]
        if len(behavior_ids) > 0:
            event_node.behavior_node_ids = behavior_ids
        return event_node
    
    def add_manual_event(self, event):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.manual_event_list) + 1
        node_id = node_count
        created = event.time
        event_node = EventNode(node_id, node_count, type_count, created, None, event, plot_id=event.plot_id)
        self.manual_event_list[0:0] = [event_node]
        self.id_to_node[node_id] = event_node
        return event_node
    
    
    def add_thought(self, thought, event_ids=[]):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.thought_list) + 1
        node_id = node_count
        created = thought.time
        thought_node = ThoughtNode(node_id, node_count, type_count, created, None, thought, plot_id=thought.plot_id)
        self.thought_list[0:0] = [thought_node]
        self.id_to_node[node_id] = thought_node
        thought_node.event_node_ids = event_ids
        
        plot = self.plot_id_to_node[thought.plot_id]
        plot.thought_node_ids[0:0] = [node_id] 
        return thought_node
        
    
    def add_behavior(self, behavior):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.behavior_list) + 1
        node_id = node_count
        created = behavior.time
        plot = self.plot_id_to_node[behavior.plot_id]

        last_self_node_id = -1
        last_partner_node_id = -1
        
        for j in range(len(plot.behavior_node_ids)): 
            if self.id_to_node[plot.behavior_node_ids[j]].behavior.self_name == behavior.self_name and last_self_node_id == -1: 
                last_self_node_id = plot.behavior_node_ids[j]
                self.id_to_node[plot.behavior_node_ids[j]].next_self_node_id = node_id
            if self.id_to_node[plot.behavior_node_ids[j]].behavior.self_name != behavior.self_name and last_partner_node_id == -1:
                last_partner_node_id = plot.behavior_node_ids[j]
                self.id_to_node[plot.behavior_node_ids[j]].next_partner_node_id = node_id
        behavior_node = BehaviorNode(node_id, node_count, type_count, created, None, behavior, 
                                     last_self_node_id=last_self_node_id, last_partner_node_id=last_partner_node_id)
        self.behavior_list[0:0] = [behavior_node]
        self.id_to_node[node_id] = behavior_node        
        plot.behavior_node_ids[0:0] = [node_id]
        return behavior_node
    
    
    def add_relationship(self, relationship, behavior_node_ids = [], event_node_ids = [], thought_node_ids = []):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = 1
        for key in self.relationship_dicts.keys():
            type_count += len(self.relationship_dicts[key])
        node_id = node_count
        created = relationship.time
        relationship_node = RelationshipNode(node_id, node_count, type_count, created, None, relationship, plot_id=relationship.plot_id)
        relationship_node.behavior_node_ids = behavior_node_ids
        relationship_node.event_node_ids = event_node_ids
        relationship_node.thought_node_ids = thought_node_ids
        
        if relationship.partner_name not in self.relationship_dicts.keys():
            self.relationship_dicts[relationship.partner_name] = []
        self.relationship_dicts[relationship.partner_name][0:0] = [relationship_node]
        
        self.id_to_node[node_id] = relationship_node
        if relationship.plot_id == -1:
            return relationship_node
        plot = self.plot_id_to_node[relationship.plot_id]
        plot.relationship_node_ids[0:0] = [node_id]
        return relationship_node
   

    def add_emotion(self, emotion):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.emotion_list) + 1
        node_id = node_count
        created = emotion.time
        emotion_node = EmotionNode(node_id, node_count, type_count, created, None, emotion, plot_id=emotion.plot_id)
        self.id_to_node[node_id] = emotion_node
        if emotion.plot_id == -1:
            return emotion_node
        self.emotion_list[0:0] = [emotion_node]
        plot = self.plot_id_to_node[emotion.plot_id]
        plot.emotion_node_ids[0:0] = [node_id]
        return emotion_node

    
    def add_core_self(self, coreself):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.core_self_list) + 1
        node_id = node_count
        created = coreself.time
        coreself_node = CoreselfNode(node_id, node_count, type_count, created, None, coreself, plot_id=coreself.plot_id)
        self.id_to_node[node_id] = coreself_node
        if coreself.plot_id == -1:
            return coreself_node
        plot = self.plot_id_to_node[coreself.plot_id]
        plot.core_self_node_ids[0:0] = [node_id]
        return coreself_node

    
    def add_motivation(self, motivation):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.motivation_list) + 1
        node_id = node_count
        created = motivation.time
        motivation_node = MotivationNode(node_id, node_count, type_count, created, None, motivation, plot_id=motivation.plot_id)
        self.id_to_node[node_id] = motivation_node
        if motivation.plot_id == -1:
            return motivation_node
        self.motivation_list[0:0] = [motivation_node]
        plot = self.plot_id_to_node[motivation.plot_id]
        plot.motivation_node_ids[0:0] = [node_id]
        return motivation_node
    
    
    def add_topic(self, topic):
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.topic_list) + 1
        node_id = node_count
        created = topic.time
        topic_node = TopicNode(node_id, node_count, type_count, created, None, topic, plot_id=topic.used_plot_id)
        self.topic_list[0:0] = [topic_node]
        self.id_to_node[node_id] = topic_node
        plot = self.plot_id_to_node[topic.used_plot_id]
        plot.topic_node_ids[0:0] = [node_id]
        return topic_node

    
    def retrieve_thoughts_by_embedding(self, embedding, topk=3):
        # use embedding scores to rank the thoughts
        scores = []
        node_ids = []
        for thought in self.thought_list:
            if not thought.thought.forgot:
                scores.append(thought.thought.get_score_from_embedding(embedding, self.current_plot_id))
                node_ids.append(thought.node_id)
                
        scores = np.array(scores)
        node_ids = np.array(node_ids)
        thought_list = []
        if len(scores) > 0:
            topk_index = np.argsort(scores)[-topk:]
            for id in topk_index:
                thought_list.append(self.id_to_node[node_ids[id]])
                self.id_to_node[node_ids[id]].last_accessed = datetime.datetime.now()
                self.id_to_node[node_ids[id]].thought.mark_accessed()
            return thought_list
        else:
            return []
        
        
    def get_events_from_plot(self, plot_id, manual=True):
        events = []
        if manual:
            for event_node in self.manual_event_list:
                if event_node.plot_id == plot_id-1:
                    events.append(event_node)
        if plot_id == -1:
            return events
        plot_node = self.plot_id_to_node[plot_id]
        for event_node_id in plot_node.event_node_ids:
            events.append(self.id_to_node[event_node_id])
        return events
    
    def get_manual_events_from_plot(self, plot_id):
        events = []
        for event_node in self.manual_event_list:
            if event_node.plot_id == plot_id:
                events.append(event_node)
        # events.sort(key=lambda x: x.event.time, reverse=True)
        return events
    
    def get_thoughts_from_plot(self, plot_id):
        thoughts = []
        for thought_node in self.thought_list:
            if thought_node.thought.plot_id == plot_id:
                thoughts.append(thought_node)
            else:
                pass
        return thoughts
    
        
    def retrieve_events_by_embedding(self, embedding, plot_id, manual=True, topk=3):
        scores = []
        node_ids = []
        # use embedding scores to rank the events
        if manual:
            all_event_list = self.event_list + self.manual_event_list
        else:
            all_event_list = self.event_list
        for event in all_event_list:
            if not event.event.forgot and event.plot_id != self.current_plot_id:
                scores.append(event.event.get_score_from_embedding(embedding, plot_id))
                node_ids.append(event.node_id)
            
        scores = np.array(scores)
        node_ids = np.array(node_ids)
        event_list = []
        if len(scores) > 0:
            topk_index = np.argsort(scores)[-topk:]
            for id in topk_index:
                event_list.append(self.id_to_node[node_ids[id]])
        else:
            return []
        return event_list
    
    def get_latest_behaviors(self, retention=6):
        ret_behaviors = []
        plot_id = -1
        current_retention_behaviors = self.behavior_list[:retention]
        for i in range(len(current_retention_behaviors)):
            if plot_id == -1:
                plot_id = current_retention_behaviors[i].plot_id
            if plot_id == current_retention_behaviors[i].plot_id:
                ret_behaviors.append(current_retention_behaviors[i])
        return ret_behaviors
            

    def get_relationships_by_partner_name(self, partner_name):
        if partner_name in self.relationship_dicts.keys():
            if len(self.relationship_dicts[partner_name]) > 0:
                return self.relationship_dicts[partner_name][0].relationship
        return None
    
    
    def get_context_behaviors(self, plot_id, context_retention=12):
        # retrieve context behaviors from the current plot
        context_behaviors = []
        if len(self.behavior_list) == 0:
            return context_behaviors
        tmp_behavior = self.behavior_list[0]
        if plot_id != tmp_behavior.plot_id:
            return context_behaviors
        while True:
            if tmp_behavior.last_partner_node_id != -1 and tmp_behavior.last_self_node_id != -1:
                if self.id_to_node[tmp_behavior.last_partner_node_id].behavior.time > self.id_to_node[tmp_behavior.last_self_node_id].behavior.time:
                    context_behaviors.append(self.id_to_node[tmp_behavior.last_partner_node_id])
                    context_behaviors.append(self.id_to_node[tmp_behavior.last_self_node_id])
                else:
                    context_behaviors.append(self.id_to_node[tmp_behavior.last_self_node_id])
                    context_behaviors.append(self.id_to_node[tmp_behavior.last_partner_node_id])
                tmp_behavior = self.id_to_node[tmp_behavior.last_self_node_id]
            elif tmp_behavior.last_partner_node_id != -1:
                context_behaviors.append(self.id_to_node[tmp_behavior.last_partner_node_id])
                break
            elif tmp_behavior.last_self_node_id != -1:
                context_behaviors.append(self.id_to_node[tmp_behavior.last_self_node_id])
                break
            else:
                break
        context_behaviors = context_behaviors[:context_retention]
        context_behaviors_nodes = context_behaviors[::-1]
        return context_behaviors_nodes


    def save_memory(self, mode='all', print_func=None, f=None, handle=None):
        """
        mode: 'all' for saving all the plots; 'plot' for saving the current plot
        print_func: print function
        """   
        if mode == 'all':
            plot_list = self.plot_list[::-1]
        elif mode == 'plot':
            plot_list = self.plot_list[0:1]
        else:
            pass
        
        for plot in plot_list:
            manual_event_count = 0
            for manual_event_node in self.manual_event_list:
                if manual_event_node.plot_id == plot.plot_id - 1:
                    manual_event_count += 1
                    if manual_event_count == 1:
                        print_func("################", f)
                        print_func(f"Manual Events", f)
                        if f is not None:
                            handle.writerow([f"Manual Events",])
                    print_func(f"Manual Event: {manual_event_node.event.get_log_description()}", f)
                    if f is not None:
                        dicts = manual_event_node.event.get_dicts()
                        if manual_event_count == 1:
                            handle.writerow(list(dicts.keys()))
                        handle.writerow(list(dicts.values()))
            
            # plot
            print_func("################", f)
            print_func(f"Plot {plot.plot_id}", f)
            print_func(f"Plot background: {plot.plot.plot_background}", f)
            print_func(f"Plot summary: {plot.plot.summary}", f)
            
            if f is not None:
                handle.writerow([f"Plot {plot.plot_id}",])
                handle.writerow([f"Plot background: {plot.plot.plot_background}",])
                handle.writerow([f"Plot summary: {plot.plot.summary}",])
            
            # topics
            print_func("################", f)
            print_func(f"Topics", f)
            if f is not None:
                handle.writerow([f"Topics",])
            
            for k, topic_node_id in enumerate(plot.topic_node_ids):
                topic = self.id_to_node[topic_node_id].topic
                print_func(f"Topic: {topic.get_log_description()}", f)
                if f is not None:
                    dicts = topic.get_dicts()
                    if k == 0:
                        handle.writerow(list(dicts.keys()))
                    handle.writerow(list(dicts.values()))
            print_func(20*"-", f)
            if f is not None:
                handle.writerow([""])
            
            print_func("################", f)
            print_func(f"Behaviors", f)
            if f is not None:
                handle.writerow([f"Behaviors",])
            
            # behaviors
            for k, behavior_node_id in enumerate(plot.behavior_node_ids[::-1]):
                behavior = self.id_to_node[behavior_node_id].behavior
                print_func(f"Behavior: {behavior.get_log_description()}", f)
                if f is not None:
                    dicts = behavior.get_dicts()
                    if k == 0:
                        handle.writerow(list(dicts.keys()))
                    handle.writerow(list(dicts.values()))
            print_func(20*"-", f)
            if f is not None:
                handle.writerow([""])

            print_func("################", f)
            print_func(f"Events", f)
            if f is not None:
                handle.writerow([f"Events",])
            # events
            for k, event_node_id in enumerate(plot.event_node_ids[::-1]):
                event = self.id_to_node[event_node_id].event
                print_func(f"Event: {event.get_log_description()}", f)
                if f is not None:
                    dicts = event.get_dicts()
                    if k == 0:
                        handle.writerow(list(dicts.keys()))
                    handle.writerow(list(dicts.values()))
            print_func(20*"-", f)
            if f is not None:
                handle.writerow([""])
        
            print_func("################", f)
            print_func(f"Thoughts", f)
            if f is not None:
                handle.writerow([f"Thoughts",])
            
            # thoughts
            for k, thought_node in enumerate(self.thought_list[::-1]):
                if thought_node.plot_id == plot.plot_id:
                    print_func(f"Thought: {thought_node.thought.get_log_description()}", f)
                    if f is not None:
                        dicts = thought_node.thought.get_dicts()
                        if k == 0:
                            handle.writerow(list(dicts.keys()))
                        handle.writerow(list(dicts.values()))
        
            print_func(20*"-", f)
            if f is not None:
                handle.writerow([""])
            
            print_func("################", f)
            print_func(f"Emotions", f)
            if f is not None:
                handle.writerow([f"Emotions",])
        
            # emotions
            for k, emotion_node in enumerate(self.emotion_list):
                if emotion_node.plot_id == plot.plot_id:
                    print_func(f"Emotion: {emotion_node.emotion.get_log_description()}", f)
                    if f is not None:
                        dicts = emotion_node.emotion.get_dicts()
                        if k == 0:
                            handle.writerow(list(dicts.keys()))
                        handle.writerow(list(dicts.values()))
            print_func(20*"-", f)
            if f is not None:
                handle.writerow([""])
            
            print_func("################", f)
            print_func(f"Relationships", f)
            if f is not None:
                handle.writerow([f"Relationships",])
        
            # relationships
            for partner_name in self.relationship_dicts.keys():
                for k, relationship_node in enumerate(self.relationship_dicts[partner_name][::-1]):
                    if relationship_node.plot_id == plot.plot_id:
                        print_func(f"Relationship: {relationship_node.relationship.get_log_description()}", f)
                        if f is not None:
                            dicts = relationship_node.relationship.get_dicts()
                            if k == 0:
                                handle.writerow(list(dicts.keys()))
                            handle.writerow(list(dicts.values()))

            print_func(20*"-", f)
            if f is not None:
                handle.writerow([""])

            print_func("################", f)
            print_func(f"Coreselfs", f)
            if f is not None:
                handle.writerow([f"Coreselfs",])
        
            # core selfs
            for k, coreself_node in enumerate(self.core_self_list):
                if coreself_node.plot_id == plot.plot_id:
                    print_func(f"Coreself: {coreself_node.coreself.get_log_description()}", f)
                    if f is not None:
                        dicts = coreself_node.coreself.get_dicts()
                        if k == 0:
                            handle.writerow(list(dicts.keys()))
                        handle.writerow(list(dicts.values()))
            print_func(20*"-", f)
            
            # motivation
            print_func("################", f)
            print_func(f"Motivations", f)
            if f is not None:
                handle.writerow([f"Motivations",])
            
            for k, motivation_node in enumerate(self.motivation_list):
                if motivation_node.plot_id == plot.plot_id:
                    print_func(f"Motivation: {motivation_node.motivation.get_log_description()}", f)
                    if f is not None:
                        dicts = motivation_node.motivation.get_dicts()
                        if k == 0:
                            handle.writerow(list(dicts.keys()))
                        handle.writerow(list(dicts.values()))
            print_func(20*"-", f)
            

    def save_teaser(self, save_dir=""):
        font = FontProperties(fname=self.config['Arial_path'], size=12)
        large_font = FontProperties(fname=self.config['Arial_path'], size=14)

        # relationship
        fig_width = 18
        fig_height = 2
        
        subplot_ids = []
        for key in self.plot_id_to_node.keys():
            subplot_ids.append(key)
        subplot_ids = sorted(subplot_ids)
        
        plot_ids = [subplot_id+1 for subplot_id in subplot_ids]
        
        relationship_node_ids = []
        for plot_id in subplot_ids:
            relationship_node_id = self.plot_id_to_node[plot_id].relationship_node_ids[0]
            relationship_node_ids.append(relationship_node_id)
        
        trusts = [self.id_to_node[id].relationship.trust for id in relationship_node_ids]
        intimacies = [self.id_to_node[id].relationship.intimacy for id in relationship_node_ids]
        supportivenesses = [self.id_to_node[id].relationship.supportiveness for id in relationship_node_ids]
        
        bar_width = 0.15
        r2 = plot_ids
        # r1 = np.arange(len(trusts))
        r1 = [x - bar_width for x in r2]
        r3 = [x + bar_width for x in r2]
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(r1, trusts, color='#E6E6FA', width=bar_width, label='Trust')        # Blue shade
        plt.bar(r2, intimacies, color='#FFC0CB', width=bar_width, label='Intimacy')  # Orange shade
        plt.bar(r3, supportivenesses, color='#87CEFA', width=bar_width, label='Supportiveness') # Green shade

        # Labeling and legend
        plt.xlabel('Sub Plot', fontproperties=large_font, fontweight='bold')
        plt.xticks(r2, plot_ids, fontproperties=font)
        plt.ylabel('Relationship', fontproperties=large_font)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.05, 1))
        plt.ylim(0, 9)
        plt.xlim(0, len(plot_ids)+0.5)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Show the graph
        plt.tight_layout()
        # plt.show()
        relationship_path = os.path.join(save_dir, 'teaser_relationship.png')
        plt.savefig(relationship_path, dpi=400, format='png')
        plt.close()
        time_ticks = []
        pleasures = []
        arousals = []
        dominances = []
        
        for plot_id in subplot_ids:
            emotion_node_ids = self.plot_id_to_node[plot_id].emotion_node_ids[::-1]
            for emotion_node_id in emotion_node_ids:
                emotion_node = self.id_to_node[emotion_node_id]
                pleasures.append(emotion_node.emotion.pleasure)
                arousals.append(emotion_node.emotion.arousal)
                dominances.append(emotion_node.emotion.dominance)
                time_ticks.append(emotion_node.emotion.plot_id + 0.25 + emotion_node.emotion.round / self.config['max_round_per_plot'])
        
        
        bar_width = 0.15 * (len(plot_ids)+1) / (len(time_ticks)+1)
        # r1 = np.arange(len(pleasures))
        r2 = time_ticks
        r1 = [x - bar_width for x in r2]
        r3 = [x + bar_width for x in r2]
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.bar(r1, pleasures, color='#FFD700', width=bar_width, label='Pleasure')        # Blue shade
        plt.bar(r2, arousals, color='#FF4500', width=bar_width, label='Arousal')  # Orange shade
        plt.bar(r3, dominances, color='#00008B', width=bar_width, label='Dominance') # Green shade

        # Labeling and legend
        # plt.xlabel('Sub Plot', fontproperties=large_font, fontweight='bold')
        plt.xticks(plot_ids, plot_ids, fontproperties=font)
        plt.ylabel('Emotion', fontproperties=large_font)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.05, 1))
        plt.ylim(0, 9)
        plt.xlim(0, len(plot_ids)+0.5)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Show the graph
        plt.tight_layout()
        # plt.show()
        relationship_path = os.path.join(save_dir, 'teaser_emotion.png')
        plt.savefig(relationship_path, dpi=400, format='png')
        plt.close()
        for subplot_id in subplot_ids:
            keywords = {}
            event_node_ids = self.plot_id_to_node[subplot_id].event_node_ids
            for event_node_id in event_node_ids:
                event_node = self.id_to_node[event_node_id]
                for keyword in event_node.event.keywords:
                    if keyword not in keywords.keys():
                        keywords[keyword] = event_node.event.poignancy
                    else:
                        keywords[keyword] += event_node.event.poignancy * 0.5

            # todo plot thought_node_ids
            thought_nodes = self.get_thoughts_from_plot(subplot_id)
            for thought_node in thought_nodes:
                for keyword in thought_node.thought.keywords:
                    if keyword not in keywords.keys():
                        keywords[keyword] = thought_node.thought.poignancy
                    else:
                        keywords[keyword] += thought_node.thought.poignancy * 0.5
            
            if len(keywords.keys()) == 0:
                continue
            
            wc = WordCloud(background_color="white", width=400, height=300, scale=2,
                max_font_size=200).generate_from_frequencies(keywords)
            plt.figure(figsize=(5, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            # plt.show()
            keywords_path = os.path.join(save_dir, 'teaser_keywords_plot_id_{}.png'.format(subplot_id+1))
            plt.savefig(keywords_path, dpi=400, format='png')
            plt.close()
        
        print("Teaser saved.")