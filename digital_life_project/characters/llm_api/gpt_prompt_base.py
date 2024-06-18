"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: gpt_prompt_base.py
Description: Base functions for generating GPT-4o prompts

This file is modified from (Author: Joon Sung Park, joonspk@stanford.edu) https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/prompt_template/gpt_structure.py .
"""

import json
from openai import OpenAI, AzureOpenAI
import os
import ast
import copy


Likert_description = "In the Likert scale range (1-9), 9 means extremely, 5 means neutral, 1 means not at all."


def ChatGPT_request_messages(messages, 
                             model="gpt-4o",
                             temperature=1.,
                             presence_penalty=0.0,
                             json_response=False): 
  try: 
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
    
    response_format = {"type": "json_object" if json_response else "text"}
    
    if type(messages) is not list:
        messages = [{"role": "user", "content": messages}]
        completion = client.chat.completions.create(
        model=model, 
        messages=messages,
        response_format=response_format,
        presence_penalty=presence_penalty,
        temperature=temperature,
        )
    else:
        completion = client.chat.completions.create(
        model=model, 
        messages=messages,
        response_format=response_format,
        presence_penalty=presence_penalty,
        temperature=temperature,
        )
    return completion.choices[0].message.content
  except: 
    print ("ChatGPT RETURN ERROR")
    return False


def find_all_positions(s, char):
    positions = []
    index = s.find(char)
    while index != -1:
        positions.append(index)
        index = s.find(char, index + 1)
    return positions


def check_string_to_double_quotes(example):
    res = list(example)
    left_index = -1
    right_index = -1
    indices = find_all_positions(example, "'")
    
    for i in indices:
        if i == 0 or i == len(example) - 1:
            continue
        if example[i] == "'" and (example[i+1] in [',','}',']','.',':','\n'] or example[i-1] in ['{','[','\n', ' ', ',']) and example[i-1] != "\"" and example[i+1] != "\"":
            if left_index == -1:
                left_index = i
            else:
                right_index = i
        if left_index != -1 and right_index != -1:
            res[left_index] = '"'
            res[right_index] = '"'
            left_index = -1
            right_index = -1
    return ''.join(res)


def response_reformat(response, example):
    prompt = f"Modify the format of the input string according to the format of the sample. " +\
        f"Note that the input content cannot be changed, but the format of the input must be consistent with that of the sample"
    prompt += f"\nExample:\n" + '' + str(example) + '\n'
    
    prompt += f"\nInput string: \n{response}\n"
    prompt += f"\nOutput the result directly, without any superfluous content.So the output should be \n"
    future = ChatGPT_request_messages(prompt)
    curr_gpt_response = future
    return curr_gpt_response


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   json_response=False): 
    if json_response:
        ret_prompt_json = f"\n\nOutput the response to the prompt above in json. {special_instruction}\n"
        ret_prompt_json += "Example output json:\n"
        ret_prompt_json += str(example_output)
        prompt_json = copy.deepcopy(prompt)
        if type(prompt_json) is list:
            prompt_json[-1]['content'] += ret_prompt_json
        else:
            prompt_json = '"""\n' + prompt_json + '\n"""\n'
            prompt_json += ret_prompt_json
    else:
        ret_prompt = f"\nOutput the response to the prompt above. {special_instruction}\n"
        ret_prompt += "In this output, all the strings should be surrounded by double quotes, not single quotes. This principle is very important.\n"
    
        ret_prompt += "Example output:\n"
        if type(example_output) is str:
            ret_prompt += '"' + str(example_output) + '"'
        else:
            ret_prompt += str(example_output)
        if type(prompt) is list:
            prompt[-1]['content'] += ret_prompt
        else:
            prompt = '"""\n' + prompt + '\n"""\n'
            prompt += ret_prompt
    

    if verbose: 
        print ("CHATGPT PROMPT:")
        print (prompt)

    for i in range(repeat): 
        try:
            if json_response:
                curr_gpt_response_json = ChatGPT_request_messages(prompt_json, json_response=True)
                curr_gpt_response = json.loads(curr_gpt_response_json)
            else:
                curr_gpt_response = ChatGPT_request_messages(prompt)
                if curr_gpt_response == False:
                    continue    
                curr_gpt_response = curr_gpt_response.strip()
                curr_gpt_response = check_string_to_double_quotes(curr_gpt_response)
            try:
                if not json_response:
                    if type(example_output) is not str:
                        curr_gpt_response = ast.literal_eval(curr_gpt_response)
                if func_validate(curr_gpt_response, prompt=prompt): 
                    return func_clean_up(curr_gpt_response, prompt=prompt)
            except:
                curr_gpt_response = response_reformat(curr_gpt_response, example_output)
                curr_gpt_response = check_string_to_double_quotes(curr_gpt_response)
                try:
                    if type(example_output) is not str:
                        curr_gpt_response = ast.literal_eval(curr_gpt_response)
                except:
                    pass
                if func_validate(curr_gpt_response, prompt=prompt): 
                    return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print ("---- repeat count: \n", i, curr_gpt_response)
                print (curr_gpt_response)
                print ("~~~~")
        except:
            pass

    return fail_safe_response()