"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: sociomind_ai_society.py
Description: Simulation for AI society application 
"""

from datetime import datetime
import os
import yaml
import argparse
from digital_life_project.autonomous_character import AutonomousCharacter


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)


def main_ai_society(config):
    characters = []
    # initialize characters
    for key in config['characters_info'].keys():
        characters.append(AutonomousCharacter(
            key, config, scene=None, place='bookshelf', has_body=False))
    
    characters[0].set_interact_partner(characters[1])
    characters[1].set_interact_partner(characters[0])
    
    # start simulation
    for _ in range(50):
        for character in characters:
            character.reaction()


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='test_marginal.yaml')
    argparser.add_argument('--replay_dir', type=str, default='')
    argparser.add_argument('--llm_service_type', type=str, default='openai', help='openai, azure')
    args = argparser.parse_args()
    
    now = datetime.now()
    formatted_time = now.strftime('%y%m%d-%H%M%S')
    save_dir = os.path.join('output', str(formatted_time))

    if args.llm_service_type in ['openai', 'azure']:
        os.environ["LLM_SERVICE_TYPE"] = args.llm_service_type
    else:
        raise ValueError(f'Unknown service type: {args.llm_service_type}')
    
    # load replay data or start new simulation
    if args.replay_dir != '':
        config_path = os.path.join(args.replay_dir, 'config.yaml')
        config = load_yaml(config_path)
        config['replay_dir'] = args.replay_dir
        file_name = args.replay_dir.split('/')[-1]
        save_dir = os.path.join('output', f'{file_name}_{formatted_time}')
    else:
        filename = args.config
        config_dir = r'.\digital_life_project\characters\configs'
        config_path = os.path.join(config_dir, filename)
        config = load_yaml(config_path)
        config['replay_dir'] = ''
        save_dir = os.path.join('output', f'{filename.split(".")[0]}_{formatted_time}')
    
    os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir
    
    # save config file
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as fid:
        yaml.dump(config, fid)

    # start simulation
    if config['application_type'] == 'AI society':
        main_ai_society(config)
    else:
        raise ValueError(f'Unknown application type: {config["application_type"]}')
    