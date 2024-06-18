"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: persona_instruct.py
Description: Define the persona instruction database class for the characters
"""
import numpy as np
from digital_life_project.characters.brain_sys.utils import *
from digital_life_project.characters.llm_api.gpt_prompt_brain import *
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError



class PersonaInstructionDatabase():
    def __init__(self, config):
        self.config = config
        self.persona_instructions = {}
        self.embeddings = {}
        
        self.check_flag = False
        if check_if_file_exists(self.config['persona_database']):
            # load database
            database = np.load(self.config['persona_database'], allow_pickle=True).item()
            self.persona_instructions = database['instructions']
            self.embeddings = database['embeddings']
        self.load_persona_instructions_from_table()
        if self.check_flag:
            self.check_embedding()
        self.rearrange_embeddings()
        print("Persona instruction database loaded!")
            

    def load_persona_instructions_from_table(self):
        if check_if_file_exists(self.config['persona_table']):
            df = pd.read_excel(self.config['persona_table'], sheet_name='data_nodupes')
            for i, row in df.iterrows():
                if i > 8000:
                    break
                if i in self.persona_instructions:
                    continue
                else:
                    self.check_flag = True
                self.persona_instructions[i] = {
                    'trait': row['label'],
                    'behavior': row['text'],
                    'key': row['key'],
                    'instrument': row['instrument'],
                    'alpha': row['alpha'],
                }
        else:
            raise Exception("persona table not found")
    
    
    def rearrange_embeddings(self):
        # trait_embeddings 
        trait_embedding_array = []
        behavior_embedding_array = []
        self.ids_list = []
        for i in self.persona_instructions.keys():
            trait_embedding_array.append(self.embeddings[self.persona_instructions[i]['trait']])
            behavior_embedding_array.append(self.embeddings[self.persona_instructions[i]['behavior']])
            self.ids_list.append(i)
        self.trait_embeddings = np.array(trait_embedding_array)
        self.behavior_embeddings = np.array(behavior_embedding_array)


    def transfer_item_to_instruction(self, item):
        extend = 'high' if item['key'] == 1 else 'low'
        desc = f"[People with {extend} {item['trait']} tend to think/behave as: {item['behavior']}]\n"
        return desc
    
    
    def retrieval_instruction_from_embeddings(self, embeddings, topk_for_each=3, topk_all=10, trait_weight=1.0, behavior_weight=1.0):
        # retrieve topk_for_each for each trait and behavior, then select topk_all from them
        select_ids = {}
        for j, embedding in enumerate(embeddings):
            trait_scores = np.dot(self.trait_embeddings, embedding)
            behavior_scores = np.dot(self.behavior_embeddings, embedding)
            overall_scores = trait_scores * trait_weight + behavior_scores * behavior_weight
            indices = np.argsort(overall_scores)[::-1][:topk_for_each]
            for index in indices:
                if self.ids_list[index] in select_ids:
                    if overall_scores[index] > select_ids[self.ids_list[index]]:
                        select_ids[self.ids_list[index]] = overall_scores[index]
                else:
                    select_ids.update({self.ids_list[index]: overall_scores[index]})
        
        sorted_ids = [key for key, value in sorted(select_ids.items(), key=lambda item: item[1], reverse=True)]
        select_ids = sorted_ids[:topk_all]
        
        instructions = []
        for id in select_ids:
            instructions.append(self.transfer_item_to_instruction(self.persona_instructions[id]))
        
        return instructions
    
    def check_embedding(self):
        for i in self.persona_instructions.keys():
            if self.persona_instructions[i]['trait'] not in self.embeddings:
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(get_embedding, self.persona_instructions[i]['trait'])
                    try:
                        self.embeddings[self.persona_instructions[i]['trait']] = np.array(future.result(timeout=30))
                    except TimeoutError:
                        print("Request timed out!")
                        self.save_database()
                        raise Exception("embedding error")
            if self.persona_instructions[i]['behavior'] not in self.embeddings:
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(get_embedding, self.persona_instructions[i]['behavior'])
                    try:
                        self.embeddings[self.persona_instructions[i]['behavior']] = np.array(future.result(timeout=30))
                        print('embedding added for item {}'.format(i)) 
                    except TimeoutError:
                        print("Request timed out!")
                        self.save_database()
                        raise Exception("embedding error")
        print("embedding checked")
    
    def save_database(self):
        np.save(self.config['persona_database'], {'instructions': self.persona_instructions, 'embeddings': self.embeddings})

if __name__ == 'main':
    config = {
    'persona_table': r"digital_life_project\characters\brain_sys\psycho_state\ipip_table.xlsx",
    'persona_database': r"digital_life_project\characters\brain_sys\psycho_state\persona_database.npy",
    }

    data_base = PersonaInstructionDatabase(config)
    data_base.save_database()
    pass