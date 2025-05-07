#!/bin/bash
# memo

PROMPT_FILE="dataset/txt_prompts/foreground.txt" 
OUTPUT_DIR="./output_original"
SCENE_TYPE="fg"

# 프롬프트 배열에 저장
mapfile -t PROMPTS < "$PROMPT_FILE"

# 프롬프트 개수 확인
TOTAL_PROMPTS=${#PROMPTS[@]}

# GPU 0 처리
(
    for ((i=0; i<$TOTAL_PROMPTS; i++)); do
        PROMPT="${PROMPTS[$i]}"  # 올바르게 변환된 프롬프트 그대로 사용

        # 파일 이름을 순차적으로 000, 001, 002
        FILENAME="$(printf "%03d" $i)"

        CUDA_VISIBLE_DEVICES=4 python attention_statistics_debug.py --prompt "$PROMPT" --filename "$FILENAME" --output_dir $OUTPUT_DIR --scene_type $SCENE_TYPE

        echo "Generated: $FILENAME"
    done
)