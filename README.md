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

python main.py --dataset PROTEINS --erase_type 1 --erase_num 1 --additional_flag
```

# Warning

Before run main code, you need to provide hugging face token and openai api key. You can find token string in **src/agent.py**

# Pipeline

If you want to run the erase process from 1 to 10, you can execute the shell script

```shell
python data_loader.py

./run.sh
```

If you find that you can not run this script, it show you this warning `Permit Deny`, you can run this command first.

```shell
chmod +x run.sh
```

# WebKG dataset execute

```shell
python WebKG_main.py --dataset cornell --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --ln --rand_split
```