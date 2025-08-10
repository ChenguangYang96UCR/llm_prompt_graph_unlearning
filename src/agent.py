import torch
from huggingface_hub import login
import os 
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from src.config import LLM, CHEKPOINTS, ROLE_DESCRIPTION, WEBKG_ROLE_DESCRIPTION
from src.utils import LOGGER
import openai
import time


login(token="")


class Agent:
    def __init__(self, 
                 agent_id: int,
                 agent_type: str,
                 llm_type: str
                ):
        
        """
        base agent class init

        Args:
            agent_id (int): agent id
            agent_type (str): agent type
            llm_type (str): llm name, such as (Llama3, deepseek)
        """
        
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.llm_type = llm_type

        # LLM 
        LOGGER.debug(f'Agent_{agent_id} ({agent_type}) initialize with llm model {llm_type}')

        #* Local model in server
        if self.llm_type in ['Llama3', 'deepseek']:
            bnb_config = self.create_bnb_config()
            model, tokenizer = self.load_model(LLM[llm_type], bnb_config)
            device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
            os.makedirs(CHEKPOINTS[llm_type], exist_ok=True)
            model.save_pretrained(CHEKPOINTS[llm_type], safe_serialization=True)
            self.model = model

            # save tokenizer for easy inference
            tokenizer = AutoTokenizer.from_pretrained(LLM[llm_type])
            tokenizer.save_pretrained(CHEKPOINTS[llm_type])
            self.tokenizer = tokenizer

        #* Call API
        else:
            pass

        # store agent prompt
        self.agent_prompt = f"{ROLE_DESCRIPTION[self.agent_type]}\n"

        # initialize agent state
        self.memory = []

    def create_bnb_config(self):

        """
        create bnb config class for llm model

        Returns:
            bnb_config: bnb config class
        """

        # ! Size change to 8 bit if gpu capacity is enough 
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True
        )
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True, 
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        return bnb_config


    def load_model(self, model_name : str, bnb_config):

        """
        load llm model

        Args:
            model_name (str): model name
            bnb_config (class): config class

        Returns:
            model: llm model
            tokenizer: llm model's tokenizer
        """        

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=True, 
            low_cpu_mem_usage = False
        )

        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    
    def clear_previous(self):

        """
        clear previous storage list
        """

        self.memory.clear()


    def ask_model(self, model, question, tokenizer):

        """
        Ask llm model to extract service list based on G-Retriever sub-graph

        Args:
            model (class): llm model
            question (string): G-Retriever sub-graph
            tokenizer (class): model tokenizer

        Returns:
            service_list: service name list
        """ 

        # full_conversation = conversation_history + [{"role": "user", "content": combined_query}]
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=2550, pad_token_id=tokenizer.eos_token_id, temperature=1, do_sample=True)
        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return model_answer
    
    def webkg_ask_openai(self, messages_list, api_key, cot):
        messages = []
        prompt_string = f""
        for message in messages_list:
            messages.append({"role": "user", "content": f"{message}\n"})
            prompt_string += f"{message}\n"
        if cot:
            final_message = "Based on all parts above, please answer the question I ask previously: must remove 10%% edges that you think it is not importan. Let's think the result step by step, and please only return the list of edges that need to be deleted on the last line (Do not use any bold, italics, code blocks, or other font formatting anywhere in your response)."
            messages.append({"role": "user", "content": final_message})
            prompt_string += f"{final_message}\n"
        else:
            final_message = "Based on all parts above, please answer the question I ask previously: must remove 10%% edges that you think it is not importan, and please only return the list of edges that need to be deleted on the last line (Do not use any bold, italics, code blocks, or other font formatting anywhere in your response)."
            messages.append({"role": "user", "content": final_message})
            prompt_string += f"{final_message}\n"

        LOGGER.debug(f"Agent prompt: {prompt_string}")
        openai.api_key = api_key
        for retry in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  # Updated to use the latest and more advanced model
                    messages=messages,
                    temperature=0.2
                )
                break
            except openai.error.ServiceUnavailableError:
                wait_time = 2 ** retry
                time.sleep(wait_time)
        result = response['choices'][0]['message']['content'].strip()
        return result

    def ask_openai(self, question, api_key):
        openai.api_key = api_key
        for retry in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  # Updated to use the latest and more advanced model
                    messages=[
                        {"role": "user", "content": question}
                    ],
                    temperature=0.2
                )
                break
            except openai.error.ServiceUnavailableError:
                wait_time = 2 ** retry
                time.sleep(wait_time)
        # st.write(response)
        result = response['choices'][0]['message']['content'].strip()
        # st.write("**Raw Model Response for Classification:**", raw_category) # make title as bold
        return result

class NormAgent(Agent):

    """
    norm agent class

    Args:
        Agent (class): base agent class

    Example:
        agent = NormAgent(1, 'MathSolver', 'deepseek')
        response = agent.get_response(query)
    """

    def __init__(self, agent_id, agent_type, llm_type):

        """
        normal agent class init function 

        Args:
            agent_id (int): agent id
            agent_type (str): agent type
            llm_type (str): llm name, such as (Llama3, deepseek)

        """

        super().__init__(agent_id, agent_type, llm_type)
        self.openai_key = ''

    def get_response(self, query : str, previous_flag : bool = False) -> str:

        """
        get llm response for mornal agent

        Args:
            query (str): user query
            previous_flag (bool, optional): use previous answer or not. Defaults to False.

        Returns:
            str: llm response
        """        

        if previous_flag:
            attempted_solution = "\n".join(self.memory)
            prompt = f"{self.agent_prompt}\n\nGraph information: {query}\n\n Attempted Solution:{attempted_solution}\n\n"
        else: 
            prompt = f"{self.agent_prompt}\n\nGraph information: {query}\n\n"
        
        LOGGER.debug(f'agent_{self.agent_id}({self.agent_type}) generated response.')

        #* Local model in server
        if self.llm_type in ['Llama3', 'deepseek']:
            response = self.ask_model(self.model, prompt, self.tokenizer)                    

            if self.llm_type == 'deepseek':
                response = response.split("</think>")[-1]
        else:
            response = self.ask_openai(prompt, self.openai_key)

        # update agent memory
        self.memory.append(response)
        return response
    
    def get_agent_type(self) -> str :
        
        """
        Get agent type

        Returns:
            str: agent type
        """

        return self.agent_type

class WebKGAgent(Agent):
    """
    WebKGAgent agent class

    Args:
        Agent (class): base agent class

    Example:
        agent = WebKGAgent(1, 'MathSolver', 'deepseek')
        response = agent.get_response(query)
    """

    def __init__(self, agent_id, agent_type, llm_type):

        """
        WebKGAgent agent class init function 

        Args:
            agent_id (int): agent id
            agent_type (str): agent type
            llm_type (str): llm name, such as (Llama3, deepseek)
        """
        super().__init__(agent_id, agent_type, llm_type)
        self.agent_prompt = f"{WEBKG_ROLE_DESCRIPTION[self.agent_type]}\n"
        self.openai_key = ''

    def get_response(self, edge : list, nodes_label : dict, face_list : list, diff_list : list, additional_flag : bool, addition_type : str, cot : bool) -> str:  

        Edge_info = f""" edge format is [node_id, node_id, edge_label], and edge list is: {edge} . Please remember these information. """
        Node_info = f""" Node label format is {{ndoe id, node label}} , and node label dict is : {nodes_label} . Please remember these information."""

        if additional_flag:
            sc_prompt = f""" And The graph simplicial complex list is : {face_list} .
                """
            
            diff_prompt = f""" And I will show you a list, these numerical values represent the total persistence, calculated as the sum of lifespans (i.e., the difference between death and birth times) of topological features associated with each node's structure in the persistence diagram.
                List: {diff_list}
                """

            if addition_type == 'sc':
                addition_prompt = sc_prompt
            
            if addition_type == 'tda':
                addition_prompt = diff_prompt

            if addition_type == 'combine':
                addition_prompt = sc_prompt + diff_prompt
        
        LOGGER.debug(f'agent_{self.agent_id}({self.agent_type}) generated response.')

        #* Local model in server
        edge_prompt = Edge_info
        if cot:
            node_prompt = f"{self.agent_prompt}\n" + "Let's think the result step by step.\n" + f"{Node_info}"
        else:
            node_prompt = f"{self.agent_prompt}\n" + f"{Node_info}"
        Final_prompt = f" I've provided all the necessary information. Please complete the analysis."

        if self.llm_type in ['Llama3', 'deepseek']:
            messages_list = []
            response = self.ask_model(self.model, node_prompt, self.tokenizer) 
            if additional_flag:
                response = self.ask_model(self.model, edge_prompt + addition_prompt + Final_prompt, self.tokenizer)        
            else:
                response = self.ask_model(self.model, edge_prompt + Final_prompt, self.tokenizer)     

            if self.llm_type == 'deepseek':
                response = response.split("</think>")[-1]
        else:
            messages_list = []
            if additional_flag:
                messages_list.append(node_prompt)
                messages_list.append(edge_prompt + addition_prompt + Final_prompt)
                response = self.webkg_ask_openai(messages_list, self.openai_key, cot)
            else:
                messages_list.append(node_prompt)
                messages_list.append(edge_prompt + Final_prompt)
                response = self.webkg_ask_openai(messages_list, self.openai_key, cot)

        # update agent memory
        self.memory.append(response)
        return response
    
    def get_agent_type(self) -> str :
        
        """
        Get agent type

        Returns:
            str: agent type
        """

        return self.agent_type

class CombineAgent(Agent):

    """
    combine agent class 

    Args:
        Agent (class): base agent class

    Example:
        norm_agent = NormAgent(1, 'MathSolver', 'deepseek')
        response = norm_agent.get_response(query)
        combine_agent = CombineAgent(2, 'combiner', 'deepseek')
        combine_response = combine_agent.get_response(combine_query, [('MathSolver' , response)])
    """

    def __init__(self, agent_id, agent_type, llm_type):

        """
        combine agent class init function 

        Args:
            agent_id (int): agent id
            agent_type (str): agent type
            llm_type (str): llm name, such as (Llama3, deepseek)

        """

        super().__init__(agent_id, agent_type, llm_type)

    def _generate_agents_response_prompt(self, agents_response: list) -> str:

        """
        generate the response's prompt of agents which need to be combined togather

        Args:
            agents_response (list): the list of agents' response need to be added into prompt

        Returns:
            str: agents' response prompt
        """

        prompt = ''
        for response in agents_response:
            user_type = response[0]
            user_response = response[1]
            agent_prompt = f'Response of {user_type} is: {user_response} \n'
            prompt += agent_prompt

        if prompt == '':
            prompt = f'There is no agent response.\n'

        return prompt

    def get_response(self, query : str, agent_responses : list) -> str:

        """
        get llm model response

        Args:
            query (str): user query
            agent_responses (list): the list of agents' response need to be added into prompt [(user_type, user_response)]

        Returns:
            str: llm model's response
        """        

        agents_response_prompt = self._generate_agents_response_prompt(agent_responses)
        prompt = f"{self.agent_prompt}\n\nQ:{query}\n\n{agents_response_prompt}"
        response = self.ask_model(self.model, prompt, self.tokenizer)                    
        LOGGER.debug(f'agent_{self.agent_id}({self.agent_type}) generated response.')
        if self.llm_type == 'deepseek':
            response = response.split("</think>")[-1]
    
        # update agent memory
        self.memory.append(response)
        return response