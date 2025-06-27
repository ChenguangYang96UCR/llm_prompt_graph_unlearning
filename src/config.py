
LLM = {
    "Llama3" : "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek" : "deepseek-ai/deepseek-math-7b-instruct"
}

CHEKPOINTS = {
    "Llama3" : "/data/ycg/ucr_work/multi_agent/results/llama/final_merged_checkpoint",
    "deepseek" : "/data/ycg/ucr_work/multi_agent/results/deepseek/final_merged_checkpoint"
}

ROLE_DESCRIPTION = {
    # TODO : Only return node id need to be erased
    "NodeEraser": 
        "You are a graph eraser expert. "
        "And I will give you a graph, please must remove one or more nodes that you think it is not important. "
        "I will show you the edge's information and node's lable. "
        "Please only return the list of node id that need to be deleted on the last line, such as: 3, 4, 15. "
        "There is another example: 5, 9, 11",

    "EdgeEraser": 
        "You are a graph eraser expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",

    "FeatureEraser": 
        "You are a graph eraser expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
}
