"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: gpt_prompt_brain.py
Description: GPT-4o prompt functions for generating brain processes

This file is modified from (Author: Joon Sung Park, joonspk@stanford.edu) https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/prompt_template/run_gpt_prompt.py .
"""
import os
from digital_life_project.characters.llm_api.gpt_prompt_base import *


def compare_types(obj1, obj2):
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, (list, tuple)):
        if len(obj1) == 0 or len(obj2) == 0:
            return False
        for item in obj1:
            if not compare_types(item, obj2[0]):
                return False
                
    elif isinstance(obj1, dict):
        if len(obj1) != len(obj2):
            return False
        for key1, key2 in zip(sorted(obj1.keys()), sorted(obj2.keys())):
            if key1 != key2:
                return False
            if not compare_types(key1, key2) or not compare_types(obj1[key1], obj2[key2]):
                return False
                
    return True


def get_embedding(text, model="text-embedding-ada-002"):
    
    llm_service_type = os.environ["LLM_SERVICE_TYPE"]
    if llm_service_type == 'openai':
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.environ["OPENAI_BASE_URL"]
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif llm_service_type == 'azure':
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version="2024-05-01-preview")
    else:
        raise ValueError("Invalid llm service")
    
    text = text.replace("\n", " ")
    if not text: 
        text = "this is blank"
    res = None
    while type(res) != list:
        try:
            response = client.embeddings.create(input=[text], model=model)
            res = response.data[0].embedding
        except:
            print("get_embedding ERROR")
    return res
        

def run_gpt_get_quantitative_emotion_from_description(desc, verbose=False):
    if verbose:
        print('run_gpt_get_quantitative_emotion_from_description: ', desc)
    prompt = f"This is about a person's emotion description: [{desc}]." +\
        f"Assume you are a very professional psychologist. Using the PAD theory of psychology and Likert scale (1-9), score on pleasure, arousal and dominance." +\
        f"The higher the score, the higher the strength." +\
            f"According to Paul Ekman's basic emotion theory, human have basic emtions: wrath, grossness, fear, joy, loneliness, shock, " +\
                f"amusement, contempt, contentment, embarrassment, excitement, guilt, pride in achievement, relief, satisfaction, sensory pleasure, and shame." +\
        f"So the result should be"
    
    example_output = {"pleasure": 5, "arousal": 5, "dominance": 5}
    
    special_instruction = f"The output must follow the format of the example output mentioned above."
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return {'pleasure': -1, 'arousal': -1, 'dominance': -1}
    
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        emotion_dicts = {"pleasure":0 , "arousal":0, "dominance":0}
        flag = True
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
            
            for key, value in emotion_dicts.items():
                if key not in gpt_dicts.keys() or type(gpt_dicts[key]) != int or gpt_dicts[key] < 1 or gpt_dicts[key] > 9:
                    flag = False
                else:
                    emotion_dicts[key] = value
        else:
            flag = False
        if flag:
            return emotion_dicts
        else:
            raise Exception("Invalid output format")
    
    if desc == '':
        return get_fail_safe()
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose, json_response=True)
    return output


def run_gpt_get_quantitative_personality_from_description(desc, verbose=False):
    if verbose:
        print('run_gpt_get_quantitative_personality_from_description: ', desc)
    prompt = f"This is about a person's personality description: [{desc}]." +\
        f"Assume you are a very professional psychologist. Using the Big Five theory of psychology and Likert scale (1-9), score on openness, conscientiousness, extraversion, agreeableness, neuroticism." +\
        f"The higher the score, the higher the strength." +\
        f"So the result should be"
    
    example_output = {'openness': 5, 'conscientiousness':5, 'extraversion':5, 'agreeableness':5, 'neuroticism':5,}
    
    special_instruction = f"The output must follow the format of the example output mentioned above."
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return {'openness': -1, 'conscientiousness':-1, 'extraversion':-1, 'agreeableness':-1, 'neuroticism':-1,}
    
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        emotion_dicts = {'openness': -1, 'conscientiousness':-1, 'extraversion':-1, 'agreeableness':-1, 'neuroticism':-1,}
        flag = True
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
            
            for key, value in emotion_dicts.items():
                if key not in gpt_dicts.keys() or type(gpt_dicts[key]) != int or gpt_dicts[key] < 1 or gpt_dicts[key] > 9:
                    flag = False
                else:
                    emotion_dicts[key] = value
        else:
            flag = False
        if flag:
            return emotion_dicts
        else:
            raise Exception("Invalid output format")
    
    if desc == '':
        return get_fail_safe()
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up,
                                            verbose, json_response=True)
    return output 


def run_gpt_get_qualitative_personality_from_quantitative(personality, verbose=False):
    if verbose:
        print('run_gpt_get_qualitative_personality_from_quantitative: ', personality)
    prompt = f"Assume you are a very professional psychologist. This is a quantitative evaluation of a person: [{personality}].\n" +\
        f"The evaluation based on the Big Five theory of psychology and Likert scale (1-9), score on openness, conscientiousness, extraversion, agreeableness, neuroticism." +\
        f"The higher the score, the higher the strength." +\
        f"Based on the score and the Goldberg's personality trait markers, describe the personality of the person.\n" +\
        f"So the description should be: "
    
    example_output = "happy, excited, and confident"
    
    special_instruction = f"The output must follow the string format of the example output. The description should not be more than 15 words."
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return 'happy, excited, and confident'
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        if gpt_response is str and len(gpt_response.split()) <= 15:
            return gpt_response
        else:
            raise Exception("Invalid output format")
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up, verbose)
    return output


def run_gpt_get_qualitative_emotion_from_quantitative(emotion_dicts, verbose=False):
    if verbose:
        print('run_gpt_get_qualitative_emotion_from_quantitative: ', emotion_dicts)
    prompt = f"Assume you are a very professional psychologist. This is a quantitative evaluation of a person: [{emotion_dicts}].\n" +\
        f"The evaluation based on the PAD theory of psychology and Likert scale (1-9), score on pleasure, arousal and dominance." +\
        f"The higher the score, the higher the strength." +\
        f"Based on the score, describe the emotion of the person.\n" +\
            f"According to Paul Ekman's basic emotion theory, human have basic emtions: wrath, grossness, fear, joy, loneliness, shock, " +\
                f"amusement, contempt, contentment, embarrassment, excitement, guilt, pride in achievement, relief, satisfaction, sensory pleasure, and shame." +\
        f"So the description should be: "
    
    example_output = "happy, excited, and confident"
    
    special_instruction = f"The output must follow the string format of the example output. The description should not be more than 15 words."
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return 'happy, excited, and confident'
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        if gpt_response is str and len(gpt_response.split()) <= 15:
            return gpt_response
        else:
            raise Exception("Invalid output format")
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up, verbose)
    return output


def run_gpt_get_emotion_from_prompt(prompt_, verbose=False):
    if verbose:
        print('run_gpt_get_emotion_from_prompt: ', prompt_)
    prompt = prompt_ +\
        f"\n\nThe emotion are described based on the PAD theory of psychology and Likert scale (1-9), score on pleasure, arousal and dominance." +\
        f"So the description should be: "
    
    example_output = {"description": "happy and peaceful", "pleasure": 8, "arousal": 3, "dominance": 6}
    
    special_instruction = f"The description should not be more than 15 words."
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return {'description': '', 'pleasure': 5, 'arousal': 5, 'dominance': 5}
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
        
        emotion_dicts = {'description': 'happy and peaceful', "pleasure":0 , "arousal":0, "dominance":0}
        if compare_types(gpt_dicts, emotion_dicts):
            return gpt_dicts
        else:
            raise Exception("Invalid output format")
        
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up,
                                            verbose, json_response=True)
    return output


def run_gpt_get_keywords_poignancy_from_description(desc, background, verbose=False):

    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
        
            if 'poignancy' in gpt_dicts.keys() and 'keywords' in gpt_dicts.keys():
                if type(gpt_dicts['poignancy']) is int and type(gpt_dicts['emergency']) is int and type(gpt_dicts['keywords']) is list:
                    return gpt_dicts
        raise Exception("Invalid output format")
        
    def get_fail_safe(): 
        return {'poignancy': 0, 'emergency': 5, 'keywords': ['meeting',]}

    def __chat_func_validate(gpt_response, prompt=""): 
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
        
    if verbose:
        print('run_gpt_get_keywords_poignancy_from_description: ', desc)

    prompt = f"Here is a brief description of event: '{desc}'\n" +\
        f"This event happened based on the background: \n---\n '{background}'\n---\n" +\
        f"Please assign the poignancy of the event and the keywords of the event.\n" +\
        f"For the poignancy, on the scale of 1 to 9, where 1 is purely mundane (e.g., wave hand, smile) and 9 is extremely poignant (e.g., a break up, love betray), rate the likely poignancy of the event." +\
        f"For emergency, it is measured based on the urgency of the event, on the scale of 1 to 9, where 1 is not urgent at all and 9 is extremely urgent.\n" +\
        f"For the keywords list, they should effectively describe the event and reveal the topic of the event. Do not output the name of the character as keyword.\n" +\
        f"So the result should be: "

    example_output = {"poignancy": 5, 'emergency': 5, "keywords": ["argue", "cry"]}
    special_instruction = "Keywords are python list of string. The output must follow the string format of the example output." ########
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __func_clean_up, verbose, json_response=True)
    return output


def run_gpt_get_quantitative_relationship_from_description(desc, attitude='', verbose=False):
    if verbose:
        print('run_gpt_get_quantitative_relationship_from_description: ', desc)
    prompt = f"This is a relationship description between two persons: '{desc}'." +\
        f"One person hold the attitude towards the another: '{attitude}'" +\
        f"Using the social psychological theory to measure the relationship in a Likert scale (1-9), score on intimacy, trust, and supportiveness." +\
        f"The higher the score, the higher the strength." +\
        f"So the result should be"
    
    example_output = {"intimacy": 5, "trust": 5, "supportiveness": 5}
    
    special_instruction = f""
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return {'intimacy': 5, 'trust': 5, 'supportiveness': 5}
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        relationship_dicts = {'intimacy': 5, 'trust': 5, 'supportiveness': 5}
        flag = True
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
            for key, value in relationship_dicts.items():
                if key not in gpt_dicts.keys() or type(gpt_dicts[key]) != int:
                    flag = False
                else:
                    relationship_dicts[key] = value
        else:
            flag = False
        if flag:
            return relationship_dicts
        else:
            raise Exception("Invalid output format")
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose, json_response=True)
    return output


def run_gpt_get_topics_from_events_personal_info(personal_info, relevant_info, verbose=False):
    if verbose:
        print('run_gpt_get_topics_from_events_personal_info.')
    prompt = personal_info + relevant_info +\
        f"Based on the information above, generate a list of topics that you would like to start to talk with your partners in the next interaction. "+\
         f"Do not generate topics similar or identical to previous used topics." +\
                f"Topics should include specific events that are consistent with the way things are going." +\
            f" The new topic should conform to the time line of the events, thoughts, and background and realistic setting of the original story." +\
                f"It should be plausible with previous events and have specific and detailed plots. Do not produce topics unrelated to characters, events, etc." +\
                    f"Each proposed topic are a python dict, in which the key 'description' means the description, 'summary' means the summary of the description within 5 words, 'emergency' and 'poignancy' are in a Likert scale to measure the emergency and importance of the topic, " +\
                    f"'partner_name' is the person's name you want to talk to.So the result should be:"
            
    example_output = [{"description": "Xiaotao is curious about AI techniques, especially GPT-5.", "summary": "curious about AI","poignancy": 6, "emergency": 2, "partner_name": "Zhixu"},
        {"description": "Xiaotao found herself as a robot, not real life.", "summary": "I'm a robot, not human","poignancy": 8, "emergency": 8, "partner_name": "Zhixu"},]
            # "{'description': 'ZHixu lost his laptop for his games.', 'poignancy': 3, 'emergency': 7, 'partner_name': 'Zhixu'}]"
    
    special_instruction = "Remember all your proposed topics should based on your personal innate traits, current events, and relevant memeories." #f"Make sure that the key of the elements (dictionary) in the python list are the same as the example, while the values should based on the content provided above."
    special_instruction += "Don't repeat the topics in the example and previous used topics." 
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
         return [{'description': '', "summary": "", 'poignancy': 1, 'emergency': 1, 'partner_name': 'Zhixu'},]
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_list = gpt_response
        if type(gpt_list) is dict:
            if len(gpt_list) == 1:
                for key in gpt_list.keys():
                    tmp_list = gpt_list[key]
                    break
                gpt_list = tmp_list
        topic_example = {'description': 'Xiaotao found herself as a robot, not real life.', 'poignancy': 3, 'emergency': 3, 'partner_name': 'Zhixu', "summary": "",}
    
        flag = True
        if type(gpt_list) is list and len(gpt_list) > 0:
            for gpt_dict in gpt_list:
                if not compare_types(gpt_dict, topic_example):
                    flag = False
        else:
            flag = False
        if flag:
            return gpt_list
        else:
            raise Exception("Invalid output format")
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            8, fail_safe, __chat_func_validate, __func_clean_up,
                                            verbose, json_response=True)
    if type(output) is list:
        for item in output:
            if len(item['summary'].split()) > 7:
                item['summary'] = run_gpt_summarize_sentence(item['description'], lens=5, verbose=verbose)
    return output


def run_gpt_get_indices_from_deduplicating_topics(prompt, topic_acc=0, verbose=False):
    if verbose:
        print('run_gpt_get_indices_from_deduplicating_topics.')
    prompt = prompt + "So the results should be: "
            
    example_output = [0,1,3]
    
    special_instruction = f"The output must follow the format of the example output mentioned above."
    special_instruction += f"{Likert_description}"
    
    def get_fail_safe(): 
        return list(range(topic_acc))
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_list = gpt_response
        if type(gpt_list) is dict:
            if len(gpt_list) == 1:
                for key in gpt_list.keys():
                    tmp_list = gpt_list[key]
                    break
                gpt_list = tmp_list
        if type(gpt_list) is list and len(gpt_list) > 0:
            for id in gpt_list:
                if type(id) != int or not 0 <= id < topic_acc:
                    raise Exception("Invalid output format")
            return gpt_list
        else:
            raise Exception("Invalid output format")
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            8, fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose, json_response=True)
    return output


def check_format_length(inputs, max_lengths):
    results  = []
    for i, input in enumerate(inputs):
        if len(input.split()) > max_lengths[i]:
            results.append(False)
        else:
            results.append(True)
    return results


def run_gpt_refinement(behavior, verbose=False):
    speech_max_length = 30
    expression_max_length = 4
    motion_max_length = 6
    place_max_length = 3
    
    def get_fail_safe(): 
        return ""
    
    def __chat_func_validate(gpt_response, prompt=""):
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        res = gpt_response
        return res
    
    output_lists = [behavior['speech'], behavior['expression'], behavior['motion'], behavior['place']]
    max_lengths = [speech_max_length, expression_max_length, motion_max_length, place_max_length]
    keys = ['speech', 'expression', 'motion', 'place']
    results = check_format_length(output_lists, max_lengths)
    for i, result in enumerate(results):
        if not result:
            prompt = f"The [{keys[i]}] of a person is [{output_lists[i]}]." +\
                f"Please simplify it by retaining core meanings and make it not more than {max_lengths[i]} words." +\
                    f"If this is a speech, simplify it the way people would in everyday speech."+\
                        f"The simplified results should be: "
            example_output = "How are you?"
            special_instruction = 'Remember to keep the core meanings.'
            fail_safe = get_fail_safe
            res_ = output_lists[i]
            for j in range(4):
                if len(res_.split()) < max_lengths[i]:
                    behavior[keys[i]] = res_
                    break
                else:
                    res_ = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 5, fail_safe,
                                                __chat_func_validate, __func_clean_up, verbose=verbose)
    return behavior


def run_gpt_summarize_sentence(sentence, lens=5, verbose=False):
    if verbose:
        print('run_gpt_summarize_sentence: ', sentence)
    prompt = f"This is a sentence: [{sentence}]." +\
        f"Summarize the sentence without losing key information in no more than {lens} words, and the result should be: "
    
    example_output = "Xiaotao is curious about AI."
    
    special_instruction = ""
    
    def get_fail_safe(): 
        return ""
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        if type(gpt_response) is str and len(gpt_response.split()) <= lens:
            return gpt_response
        else:
            raise Exception("Invalid output format")
    
    fail_safe = get_fail_safe
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 
                                            5, fail_safe, __chat_func_validate, __func_clean_up, verbose)
    return output


def run_gpt_prompt_decision_dialog(messages, names=['Xiaotao', 'Zhixu'], verbose=False):
    def get_fail_safe(): 
        return {'self_name': names[0], 'speech': '', 'expression': '', 'motion': '', 'place': '', 'partner_name': names[1],}
  
    def __chat_func_validate(gpt_response, prompt=""):
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 

    
    def __func_clean_up(gpt_response, prompt=""):
        example = {'self_name': 'Xiaotao','speech': 'How are you?', 'expression': 'smile', 'motion': 'wave hands', 'place': 'bookshelf', 'partner_name': 'Zhixu',}
        behavior_dict = {}
        if gpt_response in ['END', 'end', '"END"', '"end"']:
            return gpt_response
        for item in gpt_response.split('<'):
            if item == '':
                continue
            if '>' in item and len(item) >= 2:
                key, value = item.split('>')
                if key in example.keys():
                    behavior_dict[key] = value
        
        if compare_types(behavior_dict, example):
            behavior_dict['place'] = behavior_dict['place'].strip().lower()
            if behavior_dict['place'] not in ["sofa", "desk", "dining table", "bookshelf"]:
                raise Exception("Invalid output format")
            else:
                return behavior_dict
        else:
            raise Exception("Invalid output format")
    
    example_output = "<self_name>Xiaotao<speech>How are you?<expression>smile<motion>wave hands<place>bookshelf<partner_name>Zhixu"
    
    fail_safe = get_fail_safe
    special_instruction = f"The output must follow the format of the example output mentioned above. Remember in this interactive conversation body language weigh 30% of the signal expression." +\
     f"Note that your motion should be physics consistent and semantic variant with the previous motion, and it is best to use the motion with obvious semantics to express your own emotions and language." +\
        f"Stop being too polite and cooperative." +\
            f"Remember the motions are based on the moment when the people stand instead of sitting."
    special_instruction += f"{Likert_description}"
    
    output = ChatGPT_safe_generate_response(messages, example_output, special_instruction, 6, fail_safe, __chat_func_validate, __func_clean_up, verbose=verbose)

    if output in ['END', 'end', '"END"', '"end"']:
        return output

    if output != False: 
        output = run_gpt_refinement(output, verbose=verbose)
        return output


def run_gpt_prompt_summarize_events_from_dialog(innate_trait_prompt, memory_prompt, verbose=False):
    def get_fail_safe(): 
        return []
  
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 
    
    def __func_clean_up(gpt_response, prompt=""):
        event_list = gpt_response
        flag = True
        event_example_dicts = {'description': 'TODO', 'keywords': ['TODO'], 'poignancy': 5, 'emergency': 5}
        if type(event_list) is dict:
            if len(event_list) == 1:
                for key in event_list.keys():
                    tmp_list = event_list[key]
                    break
                event_list = tmp_list
        if type(event_list) is list:
            for event in event_list:
                if not compare_types(event, event_example_dicts):
                    flag = False
        else:
            flag = False
        if flag:
            return event_list
        else:
            raise Exception("Invalid output format")
    
    if verbose:
        print('run_gpt_prompt_summarize_events_from_dialog.')
    
    example_output = [{"description":"Zhixu debates with xiaotao on fiction movies.", "keywords" : ["debates", "fiction movie"], "poignancy": 6, 'emergency': 5,}, 
                      {"description":"Zhixu knows that Xiaotao pretend to like fiction movies ", "keywords" : ["pretend"], "poignancy": 5, 'emergency': 3,},]
    
    fail_safe = get_fail_safe
    special_instruction = "The output must continue the sentence with any other words except the proposed format. "
    special_instruction += f"Attention: the number of summarized events should not be more than 3."
    
    prompt = f"{innate_trait_prompt} \n {memory_prompt} "+\
        f"Based on the dialog above, summarize the key events and output the event list in python list format." +\
            f"Each event are in a python dict formation, in which the key 'description' means the description," +\
                f"'poignancy' is in a Likert scale (1-9) to measure the importance of the event," +\
                    f"'emergency' is in a Likert scale (1-9) to measure the urgency of the event," +\
                    f" 'keywords' is a list of keywords that effectively describe the event and reveal the topic, do not include the name of the persons." +\
                        f"The events summarized should contain effective social information and be summarized according to the personality, motivation, relationship, and core belief of the person." +\
                            f"The summary of events must not be too detailed, must be a general summary of what happened, and do not repeat between events." +\
                            f"So the result should be: "
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 6, fail_safe,
                                            __chat_func_validate, __func_clean_up, verbose=verbose, json_response=True)
    if output != False: 
        return output


def run_gpt_prompt_summarize_thoughts_from_events(innate_trait_prompt, memory_prompt, verbose=False):
    def get_fail_safe(): 
        return []
  
    def __chat_func_validate(gpt_response, prompt=""):
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False 

    def __func_clean_up(gpt_response, prompt=""):
        thought_list = gpt_response
        flag = True
        thought_example_dicts = {'description': 'TODO', 'poignancy': 5, 'keywords': ['TODO']}
        if type(thought_list) is dict:
            if len(thought_list) == 1:
                for key in thought_list.keys():
                    tmp_list = thought_list[key]
                    break
                thought_list = tmp_list
        if type(thought_list) is list:
            for thought in thought_list:
                if not compare_types(thought, thought_example_dicts):
                    flag = False
        else:
            flag = False
        if flag:
            return thought_list
        else:
            raise Exception("Invalid output format")
    
    if verbose:
        print('run_gpt_prompt_summarize_thoughts_from_events.')
    
    example_output = [{"description":"Zhixu seems to be a straightforward person", "keywords":["straightforward"], "poignancy": 3}, {"description":"I must leave this place", "keywords": ["leave", "place"], "poignancy": 7}]
    
    fail_safe = get_fail_safe
    special_instruction = "The output must continue the sentence with any other words except the proposed format. " + \
    "Focus on the thoughts you get from the new events, rather than the thoughts you already have."
    special_instruction += f"Attention: the number of summarized thoughts should not be more than 3."
    
    prompt = f"{innate_trait_prompt} \n {memory_prompt} "+\
        f"Based on the relevant memories about relevant events and thoughts, summarize the new thoughts by current occured events and output the thought list in python list format." +\
            f"Thoughts are a bunch of minds and feelings about what's happening, and what you're picking up about what's happening." +\
            f"Each thought are in a python dict formation, in which the key 'description' means the description," +\
                f"'poignancy' is in a Likert scale (1-9) to measure the importance of the thought." +\
                    f" 'keywords' is a list of keywords that effectively describe the event and reveal the topic, do not include the name of the persons." + \
                    f"The thoughts should be summarized according to the personality, motivation, relationship, and core belief of the person, and do not repeat between thoughts." +\
                            f"So the result should be: "
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 6, fail_safe, 
                                            __chat_func_validate, __func_clean_up, verbose=verbose, json_response=True)

    if output != False: 
        return output
    
    
def run_gpt_prompt_core_self_update(innate_trait_prompt, reflection_prompt, verbose=False):
    def get_fail_safe(): 
        return {"conflict": "N", "belief": "I am a realistic human that can feel and enjoy the life."}
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False
    
    def __func_clean_up(gpt_response, prompt=""):
        if type(gpt_response) is dict:
            if len(gpt_response) == 1:
                for key in gpt_response.keys():
                    tmp_dicts = gpt_response[key]
                    break
                gpt_response = tmp_dicts
        if type(gpt_response) is dict:
            if 'conflict' in gpt_response.keys() and 'belief' in gpt_response.keys():
                if type(gpt_response['conflict']) is str and type(gpt_response['belief']) is str:
                    if gpt_response['conflict'] in ['Y', 'N']:
                        return gpt_response
        else:
            raise Exception("Invalid output format")
        
        raise Exception("Invalid results")
    
    if verbose:
        print('run_gpt_prompt_core_self_update.')
    
    example_output = {"conflict": "N", "belief": "I am a realistic human that can feel and enjoy the life."}
    fail_safe = get_fail_safe
    special_instruction = "The output must continue the sentence with any other words except the proposed python dict format. "
    prompt = innate_trait_prompt + reflection_prompt +\
                f"'poignancy' is in a Likert scale (1-9) to measure the importance of the thought. Belief is a string description, not list of thoughts." +\
                    f"So the updated core self should be "
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 6, 
                                            fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose=verbose, json_response=True)
    if output != False:
        return output
    

def run_gpt_prompt_relationship_update(innate_trait_prompt, reflection_prompt, verbose=False):
    def get_fail_safe(): 
        return {"intimacy": -1, 'trust':-1, "supportiveness":-1, 'description': "", "attitude": ""}
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
        relationship_dicts = {"intimacy": 2, 'trust':2, "supportiveness":2, 'description': "they are strangers.", 'attitude': 'neutral'}
        if compare_types(gpt_dicts, relationship_dicts):
            return gpt_dicts
        else:
            raise Exception("Invalid output format")
    
    if verbose:
        print('run_gpt_prompt_relationship_update.')
    
    example_output = {"intimacy": 2, "trust":2, "supportiveness":2, "description": "they are strangers.", "attitude": "neutral"}
    
    fail_safe = get_fail_safe
    special_instruction = ""
    special_instruction += f"{Likert_description}"
    
    prompt = innate_trait_prompt + reflection_prompt +\
                f"Trust, intimacy, and supportness is in a Likert scale (1-9) to measure the relationship between two persons." +\
                    f"So based on the new events, your new relationship with the person should be "
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 5, 
                                            fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose=verbose, json_response=True)
    if output != False:
        return output
    

def run_gpt_prompt_motivation_update(innate_trait_prompt, reflection_prompt, verbose=False):
    def get_fail_safe(): 
        return {"long_term": {"changed": "N", "value": "Explore the meaning of human life."}, "short_term": {"changed": "Y", "value": "Know Zhixu's heart"}}
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
        flag_count = 0
        if type(gpt_dicts) is dict and list(gpt_dicts.keys()) == ['long_term', 'short_term']:
            for key in gpt_dicts.keys():
                if type(gpt_dicts[key]) is dict and list(gpt_dicts[key].keys()) == ['changed', 'value']:
                    if gpt_dicts[key]['changed'] in ['Y', 'N'] and type(gpt_dicts[key]['value']) is str:
                        flag_count += 1   
        if flag_count == 2:
            return gpt_dicts
        else:
            raise Exception("Invalid output format")
    
    if verbose:
        print('run_gpt_prompt_motivation_update.')
    
    example_output = {"long_term": {"changed": "N", "value": "Explore the meaning of human life."}, "short_term": {"changed": "Y", "value": "Know Zhixu's heart"}}
    
    fail_safe = get_fail_safe
    special_instruction = ""
    
    prompt = innate_trait_prompt + reflection_prompt
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 5, 
                                            fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose=verbose, json_response=True)
    if output != False:
        return output
    
    
def run_gpt_get_plot_proposals_from_topics_and_personal_info(prompt_personal_info, topic_prompt, plot_prompt, lens=100, verbose=False):
    def get_fail_safe(): 
        return []
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False
    
    def __func_clean_up(gpt_response, prompt=""):
        plot_list = gpt_response
        if type(plot_list) is dict:
            if len(plot_list) == 1:
                for key in plot_list.keys():
                    tmp_list = plot_list[key]
                    break
                plot_list = tmp_list
        flag = True
        plot_example_dicts = {'topic_ids': [0, 2], 'poignancy': 8, 'emergency': 6, 'plot_background': 'Xiaotao knew that she is not a real life.', "summary": "Xiaotao realized she's unreal."}
        if type(plot_list) is list:
            for plot in plot_list:
                if not compare_types(plot, plot_example_dicts):
                    flag = False
            for index in plot['topic_ids']:
                if index < 0 or index > lens-1:
                    flag = False
        else:
            flag = False
        if flag:
            return plot_list
        else:
            raise Exception("Invalid output format")
    
    if verbose:
        print('run_gpt_get_plot_proposals_from_topics_and_personal_info.')
    
    example_output = [{"topic_ids": [0, 2], "poignancy": 8, 'emergency': 6, "plot_background": "Xiaotao knew that she is not a real life.", "summary": "Xiaotao realized she's unreal."},
                      {"topic_ids": [1,], "poignancy": 3, 'emergency': 4, "plot_background": "Xiaotao know that Zhixu likes reading books.", "summary": "Zhixu enjoys reading, Xiaotao knows."},]
    
    fail_safe = get_fail_safe
    special_instruction = ""
    special_instruction += f"{Likert_description}"
    
    prompt = prompt_personal_info + topic_prompt + plot_prompt +\
                f"Based on the information above, generate a list of plot proposals that he/she would like to start: "
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 5, 
                                            fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose=verbose, json_response=True)
    
    if output != False:
        for item in output:
            if len(item['summary'].split()) > 7:
                item['summary'] = run_gpt_summarize_sentence(item['plot_background'], lens=5, verbose=verbose)
        return output


def run_gpt_get_plot_setup_from_personal_info_and_background(prompt_personal_info, prompt_new_plot_info, names=['Zhixu', 'Xiaotao'], verbose=False):
    def get_fail_safe(): 
        return {names[0]: {'emotion': '', 'behavior': {'place':'chair', 'motion': ''}},
            names[1]: {'emotion': '', 'behavior': {'place':'desk', 'motion': ''}},}
    
    def __chat_func_validate(gpt_response, prompt=""): ############
        try: 
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False
    
    def __func_clean_up(gpt_response, prompt=""):
        gpt_dicts = gpt_response
        if type(gpt_dicts) is dict:
            if len(gpt_dicts) == 1:
                for key in gpt_dicts.keys():
                    tmp_dicts = gpt_dicts[key]
                    break
                gpt_dicts = tmp_dicts
        plot_dicts = {names[0]: {"emotion": "happy and curious", "behavior": {"place":"chair", "motion": "sit on the chair"}},
            names[1]: {"emotion": "thrilled and curious", "behavior": {"place":"desk", "motion": "appear near the desk"}},}
        if compare_types(gpt_dicts, plot_dicts):
            return gpt_dicts
        else:
            raise Exception("Invalid output format")
    
    if verbose:
        print('run_gpt_get_plot_setup_from_personal_info_and_background.')
    
    example_output = {names[0]: {"emotion": "happy and curious", "behavior": {"place":"chair", "motion": "sit on the chair"}},
            names[1]: {"emotion": "thrilled and curious", "behavior": {"place":"desk", "motion": "appear near the desk"}},}
    fail_safe = get_fail_safe
    special_instruction = "The output must continue the sentence with any other words except the proposed format. "
    special_instruction += f"{Likert_description}"
    
    prompt = prompt_personal_info + prompt_new_plot_info +\
                f"You should output the emotion and behavior of each person in the plot. Besides, the behavior includes the place and motion of the person." +\
                    f"Remember that the motion must be details and various like a character in an animated movie." +\
                    f"According to Paul Ekman's basic emotion theory, human have basic emtions: wrath, grossness, fear, joy, loneliness, shock, " +\
                        f"amusement, contempt, contentment, embarrassment, excitement, guilt, pride in achievement, relief, satisfaction, sensory pleasure, and shame." +\
                    f"So the result should be: "
    
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 5, 
                                            fail_safe, __chat_func_validate, __func_clean_up, 
                                            verbose=verbose, json_response=True)
    
    if output != False:
        return output