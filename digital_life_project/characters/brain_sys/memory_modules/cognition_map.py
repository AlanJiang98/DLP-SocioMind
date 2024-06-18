"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: cognition_map.py
Description: Define the cognition map class of the memory
"""
import copy


class PlaceNode:
    def __init__(self, name, node_id=-1, parent=None, children=[], neighbors=[], description='', relevant_event_ids=[]):
        self.name = name 
        self.node_id = node_id
        self.parent_id = None
        self.children_ids = []
        self.neighbor_ids = []
        self.description = ''
        self.relevant_event_ids = []


class CognitionMap:
    def __init__(self, config):
        self.config = config
        self.place_id_to_node = {}
        self.key_to_node_id = {}
        self.place_dicts = config['place_dicts']
        self.build_cognition_map()
    
    def init_cognition_map_pre(self, dicts):
        if dicts == {}:
            return
        else:
            for key in dicts.keys():
                if key not in self.key_to_node_id.keys():
                    node_id = len(self.place_id_to_node)
                    node = PlaceNode(name=key, node_id=node_id)
                    self.place_id_to_node[node_id] = node
                    self.key_to_node_id[key] = node_id
                    self.init_cognition_map_pre(dicts[key])
    
    def init_cognition_map_layer(self, dicts):
        if dicts == {}:
            return
        else:
            keys = dicts.keys()
            neighbor_ids = [node.node_id for node in self.place_id_to_node.values() if node.name in keys]
            for key in dicts.keys():
                node_id = self.key_to_node_id[key]
                node = self.place_id_to_node[node_id]
                node.parent_id = self.key_to_node_id[key]
                node.neighbor_ids = copy.deepcopy(neighbor_ids).remove(node_id)
                node.children_ids = [self.key_to_node_id[child_key] for child_key in dicts[key].keys()]
                self.init_cognition_map_layer(dicts[key])
    
    def build_cognition_map(self):
        self.init_cognition_map_pre(self.place_dicts)
        self.init_cognition_map_layer(self.place_dicts)
        
    
    def check_place(self, place_name):
        if place_name in self.key_to_node_id.keys():
            return True
        else:
            return False
    
    def get_place_id(self, place_name):
        return self.key_to_node_id[place_name]
    
    def get_place_node(self, place_name):
        return self.place_id_to_node[self.key_to_node_id[place_name]]
    
    def get_place_proposal_prompt(self, prompt="", place="room", level=3, event_node_list=[]):
        if level < 0:
            return prompt
        else:
            node = self.get_place_node(place) 
            if node.description != '':
                prompt += f"{node.name} is a place described as: {node.description}.\n"
            if len(node.children_ids) != 0:
                prompt += f"{node.name} has {len(node.children_ids)} sub-places: "
                for child_id in node.children_ids:
                    child_node = self.place_id_to_node[child_id]
                    prompt += f"{child_node.name}, "
                prompt += "\n"
            for child_id in node.children_ids:
                child_node = self.place_id_to_node[child_id]
                prompt = self.get_place_proposal_prompt(prompt, place=child_node.name, level=level-1)
        return prompt
    
                