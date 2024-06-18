"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: autonomous_character.py
Description: Autonomous character class for digital life project
"""
import os
import sys
from copy import deepcopy
import gc
sys.path.append(os.path.join(os.path.dirname(__file__)))
from digital_life_project.characters.sociomind import SocioMind


class AutonomousCharacter():
    def __init__(self,
                 name,
                 config=None,
                 expression=None,
                 motion=None,
                 place=None,
                 scene=None,
                 has_brain=True,
                 has_body=False,
                 has_eyes=False
                 ):
        """
        Autonomous Character class
        """
        self.name = name

        # activate key modules
        self.has_brain = has_brain
        self.has_body = has_body
        self.has_eyes = has_eyes

        ## initialize SocioMind system
        if self.has_brain:
            self.brain = SocioMind(name, config, character=self)
        
        # interactive partners
        self.partners = []

        # register self to scene
        self.scene = scene
        
        self.current_sensing_info = []
        self.current_observed_info = None


    def set_behavior(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.current_behavior, key, value)


    # set the interact partner into brain system
    def set_interact_partner(self, character):
        if type(character) is not list:
            self.partners = [character, ]
        else:
            self.partners = character


    def get_current_observed_info_hook(self):
        # Get its partners's behavior
        # Now it is only for inter-topic talk
        current_observed_info = {}
        try:
            for partner in self.partners:
                observed_info = partner.brain.get_observed_info()
                if self.name in observed_info.keys():
                    current_observed_info[partner.name] = observed_info[self.name]
            return current_observed_info
        except:
            return {}


    def sensing(self):
        observed_info = self.get_current_observed_info_hook()
        return observed_info


    def execute(self):
        # step 1: pass brain info into avatar
        current_decision_instruction = self.brain.psycho_state.current_behavior
        
        # step 2: execute the instructions
        # motion synthesis
        behavior_dict = current_decision_instruction.get_dicts()
        if self.has_body:
            self.behave(**behavior_dict)
            return behavior_dict


    def behave(self, speech=None, expression=None, motion=None, place=None, **kwargs):
        """
        The character will behave according to the given parameters,
        if not given, the character will behave autonomously
        :param speech: speech of the character
        :param expression: expression description
        :param motion: motion description
        :param place: place description
        :param set_passive_behave: suggest interactee's behavior
        :return: behavior message in the designated format
        """
        pass


    def passive_behave(self, behavior=None, motion=None, active_motion=None):
        """ In active-passive scheme """
        pass


    def move(self):
        """ walking, running, sitting, lying down etc """
        pass


    # current implementation is far from elegant
    def align_move(self):
        """ align move motion length, the character with shorter motion is filled with idle motion """
        pass


    def synthesize_full_motion(self, ret_motion_log=False, turn_to_face_interactee=False):
        """ Full motion = walking + behavior + necessary idle infilling motion """
        pass


    # Core step for reaction system
    def reaction(self):
        gc.collect()
        if self.brain.psycho_state.plot_state == 'end':
            return
        # step 1: sensing
        # gather information from the scene and the interaction partner
        sensing_info = self.sensing()
        if sensing_info != {}:
            # we only have one partner here
            partner_name = self.partners[0].name
            if partner_name not in self.brain.psycho_state.perserved_observed_info or \
                    sensing_info[partner_name] != self.brain.psycho_state.perserved_observed_info[partner_name]:
                self.brain.psycho_state.perserved_observed_info[partner_name] = deepcopy(sensing_info[partner_name])
            else:
                pass

        if self.brain.psycho_state.plot_state == 'working':
            # step 2: perception, memory query, decision, and execution
            # if the partner is still in an interative state, keep interacting
            # else end the plot and psychological reflection
            if sensing_info[partner_name]['plot_state'] == 'working':
                print('===', self.name, 'reaction', 'working', 'partner working')
                self.brain.perception()
                self.brain.memory_query()
                self.current_decision_instruction = self.brain.decision()
                self.brain.reflection()
                return self.execute()
            else:
                print('===', self.name, 'reaction', 'working')
                self.brain.psycho_state.plot_state = 'plot_finished'
                self.brain.reflection()
        elif self.brain.psycho_state.plot_state == 'plot_finished':
            # if this plot ends, save the logs and generate new plot proposals
            print('===', self.name, 'plot_finished')
            self.brain.end_plot()
            self.brain.save_current_plot_state()
            self.log(save_dir=self.brain.config['save_dir'], mode='all')
            self.brain.plan_plot_proposals()
            return
        elif self.brain.psycho_state.plot_state == 'plan_plot_proposals':
            # if the plot has proposes new plot candiates, start a new plot
            print('===', self.name, 'plan_plot_proposals')
            self.brain.plan_start_new_plot()
            return


    # log the brain memories
    def log(self, save_dir=None, mode='plot'):
        if save_dir is None:
            logs_path = None
        else:
            logs_path = os.path.join(save_dir, self.name + '_logs.txt')
        self.brain.save_brain_memories(logs_path, mode)
        
