# llm_prompt_graph_unlearning


```shell
conda create --name llm-prompt-graph-unlearning python=3.9 --y

conda activate llm-prompt-graph-unlearning

pip install -r requirments.txt

python data_loader.py

python main.py --erase_type 0 --erase_num 1

python main.py --erase_num 1 --latent

python main.py --erase_num 1 --additional_flag

python main.py --erase_type 1 --erase_num 1

python main.py --erase_type 1 --erase_num 1 --additional_flag
```
ß
# Warningß

Before run main code, you need to provide hugging face token and openai api key. You can find token string in **src/agent.py**