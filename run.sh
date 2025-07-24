#!/bin/bash
DATA_SET="PROTEINS"
NODE_ERASE=0
EDGE_ERASE=1
FEATURE_ERASE=2
SOURCE_FILE="multi_agent.log"
TARGET_DIR="./store/log/$DATA_SET/"

ERASE_TYPE=$NODE_ERASE

mkdir -p $TARGET_DIR

YEAR=$(date +"%Y")
MONTH=$(date +"%m")
DAY=$(date +"%d")
HOUR=$(date +"%H")

for ERASE_NUM in {5..5}; do 
    echo "Run $DATA_SET Erase Number $ERASE_NUM"
    python main.py --dataset "$DATA_SET" --erase_type $ERASE_TYPE --erase_num $ERASE_NUM 
    if [ "$ERASE_TYPE" == "$NODE_ERASE" ]; then
        FINAL_DIR="./store/log/$DATA_SET/Node"
        mkdir -p "$FINAL_DIR"
        mv "$SOURCE_FILE" "$FINAL_DIR/erase_$ERASE_NUM-$YEAR-$MONTH-$DAY-$HOUR.log"
    fi    

    if [ "$ERASE_TYPE" == "$EDGE_ERASE" ]; then
        FINAL_DIR="./store/log/$DATA_SET/Edge"
        mkdir -p "$FINAL_DIR"
        mv "$SOURCE_FILE" "$FINAL_DIR/erase_$ERASE_NUM-$YEAR-$MONTH-$DAY-$HOUR.log"
    fi 
done