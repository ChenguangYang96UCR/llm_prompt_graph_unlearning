import torch
from huggingface_hub import login
import os 
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from src.config import LLM, CHEKPOINTS, ROLE_DESCRIPTION
from src.utils import LOGGER
import openai


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
            os.makedirs(CHEKPOINTS[agent_type], exist_ok=True)
            model.save_pretrained(CHEKPOINTS[agent_type], safe_serialization=True)
            self.model = model

            # save tokenizer for easy inference
            tokenizer = AutoTokenizer.from_pretrained(LLM[llm_type])
            tokenizer.save_pretrained(CHEKPOINTS[agent_type])
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
    
    def ask_openai(self, question, api_key):
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Updated to use the latest and more advanced model
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )
        
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