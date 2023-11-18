# maskedSATD

To execute the program for the corresponding sub-research questions use the following commands: 
MAT vs. MAT
```
python main.py \
    --model microsoft/codebert-base-mlm \
    --pretrained True \
    --epochs 10 \
    --learning-rate 5e-5 \
    --max-length 512 \
    --batch-size 16 \
    --warmup-steps 100 \
    --weight-decay 0.01 \
    --comments-only True \
    --train True \
    --test True \
    --patterns MAT \
    --base-patterns MAT \
    --delete-model False \
    --path ./data \
    --evaluation-strategy accuracy
```
MAT vs. PS
```
python main.py \
    --model microsoft/codebert-base-mlm \
    --pretrained True \
    --epochs 10 \
    --learning-rate 5e-5 \
    --max-length 512 \
    --batch-size 16 \
    --warmup-steps 100 \
    --weight-decay 0.01 \
    --comments-only True \
    --train True \
    --test True \
    --patterns SATD \
    --base-patterns MAT \
    --delete-model False \
    --path ./data \
    --evaluation-strategy accuracy
```
MAT vs. Others
```
python main.py \
    --model microsoft/codebert-base-mlm \
    --pretrained True \
    --epochs 10 \
    --learning-rate 5e-5 \
    --max-length 512 \
    --batch-size 16 \
    --warmup-steps 100 \
    --weight-decay 0.01 \
    --comments-only True \
    --train True \
    --test True \
    --patterns Confusing \
    --base-patterns MAT \
    --delete-model False \
    --path ./data \
    --evaluation-strategy accuracy
```
