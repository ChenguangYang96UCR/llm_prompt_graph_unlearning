#!/bin/bash
# DATA_SET=("cornell" "texas" "wisconsin")
DATA_SET=("texas")
# cornell
# texas
# wisconsin 

NODE_ERASE=0
EDGE_ERASE=1
FEATURE_ERASE=2
SOURCE_FILE="multi_agent.log"
ERASE_TYPE=$EDGE_ERASE

YEAR=$(date +"%Y")
MONTH=$(date +"%m")
DAY=$(date +"%d")
HOUR=$(date +"%H")
MINUTES=$(date +"%M")
for item in "${DATA_SET[@]}"; do 
    TARGET_DIR="./store/log/$item/"
    mkdir -p $TARGET_DIR
    for ERASE_NUM in {1..1}; do 
        echo "Run $item Erase Number $ERASE_NUM"
        python WebKG_main.py --dataset "$item" --erase_type $ERASE_TYPE --erase_num $ERASE_NUM --hidden_channels 64 --epochs 5000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.2 --ln --rand_split --additional_flag --addition_type tda --cot --latent
        # python WebKG_main.py --dataset "$item" --erase_type $ERASE_TYPE --erase_num $ERASE_NUM --hidden_channels 32 --epochs 5000 --lr 0.0001 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.2 --ln --rand_split 
        if [ "$ERASE_TYPE" == "$NODE_ERASE" ]; then
            FINAL_DIR="./store/log/$item/Node"
            mkdir -p "$FINAL_DIR"
            mv "$SOURCE_FILE" "$FINAL_DIR/erase_$ERASE_NUM-$YEAR-$MONTH-$DAY-$HOUR-$MINUTES.log"
        fi    

        if [ "$ERASE_TYPE" == "$EDGE_ERASE" ]; then
            FINAL_DIR="./store/log/$item/Edge"
            mkdir -p "$FINAL_DIR"
            mv "$SOURCE_FILE" "$FINAL_DIR/erase_$ERASE_NUM-$YEAR-$MONTH-$DAY-$HOUR-$MINUTES.log"
        fi 
    done
done