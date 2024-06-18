"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: core_self.py
Description: Define the core self class of SocioMind
"""
from digital_life_project.characters.llm_api.gpt_prompt_brain import get_embedding
import numpy as np


class Coreself:
    def __init__(self, self_name='', central_belief=[] , features=[], plot_id: int =-1, round: int =-1, time=None):
        self.self_name = self_name
        self.plot_id = plot_id
        self.round = round
        self.central_belief = central_belief
        self.time = time
        self.features = features
        self.feature_embeddings = {}
        for feature in self.features:
            self.get_feature_embedding_from_feature(feature)
        self.get_core_self_embedding()
    
    def get_feature_embedding_from_feature(self, feature):
        if feature not in self.feature_embeddings:
            self.feature_embeddings[feature] = np.array(get_embedding(feature))
        return self.feature_embeddings[feature]
    
    def get_full_description(self):
        return f"<self_name>{self.self_name}<central_belief>{self.central_belief}" +  \
            f"<plot_id>{self.plot_id}<round>{self.round}<time>{self.time}"
    
    def get_log_description(self):
        return f"<self_name>{self.self_name}<central_belief>{self.central_belief}" +  \
            f"<plot_id>{self.plot_id}"
    
    def get_prompt_core_self(self):
        prompt = f"{self.self_name}'s central belief is: [{self.central_belief}]."
        return prompt
    
    def get_prompt_core_self_with_features(self, features):
        prompt = self.get_prompt_core_self()
        if len(features) != 0:
            prompt += f"{self.self_name}'s personal features are: [{features}]."
        return prompt
    
    def get_core_self_embedding(self):
        self.embedding = np.array(get_embedding(self.get_prompt_core_self()))
        return self.embedding
    
    def retrieval_features(self, desc_embedding, topk=1, threshold=0.9):
        # retrieval topk features over the threshold from the desc_embedding
        if len(self.feature_embeddings) != 0:
            assert len(desc_embedding) == len(self.feature_embeddings[list(self.feature_embeddings.keys())[0]])
            feature_embeddings = np.array(list(self.feature_embeddings.values()))
            feature_embeddings = feature_embeddings.reshape(feature_embeddings.shape[0], -1)
            desc_embedding = desc_embedding.reshape(1, -1)
            similarity = np.dot(desc_embedding, feature_embeddings.T)[0]
            sorted_idx = np.argsort(similarity)[::-1]
            return [self.features[idx] for idx in sorted_idx[:topk] if similarity[idx] > threshold]
        else:
            return []
    
    def get_prompt_core_self_and_features_from_embedding(self, embedding=None, topk=4, threshold=0.9):
        if type(embedding) is np.array:
            features = self.retrieval_features(embedding, topk=topk, threshold=threshold)
        elif type(embedding) is list:
            features = []
            for embedding_item in embedding:
                features.extend(self.retrieval_features(embedding_item, topk=topk, threshold=threshold))
                features = list(set(features))
        else:
            features = self.features
        embeddings = [self.feature_embeddings[feature] for feature in features]
        return self.get_prompt_core_self_with_features(features), embeddings
        
    def add_features(self, features):
        self.features.extend(features)
        for feature in features:
            self.get_feature_embedding_from_feature(feature)
    
    def get_dicts(self):
        return {
            'self_name': self.self_name,
            'central_belief': self.central_belief,
            'plot_id': self.plot_id,
            'round': self.round,
            'time': self.time,
        }
    
