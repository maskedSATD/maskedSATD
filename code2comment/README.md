
To execute the program for the corresponding research question use the following commands: 
Summarization of functions

```
python ../run.py \
  --do_train \
  --do_eval \
  --do_test \
  --model_type roberta \
  --model_name_or_path microsoft/codebert-base \
  --train_filename <path>/all/train.csv \
  --dev_filename <path>/all/val.csv \
  --test_filename <path>/all/test_sample.csv \
  --output_dir model/all_both \
  --max_source_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 5e-5 \
  --train_steps 15000 \
  --num_train_epochs 10 \
  --eval_steps 1000 \
  --save_path_prefix ""
```

The base code was taken from [https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl) and modified by the author. So that it fits the needs for the experiment.