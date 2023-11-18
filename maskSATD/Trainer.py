from transformers import DataCollatorForLanguageModeling, \
    Trainer as TransformerTrainer, \
    TrainingArguments

from Dataconsumer import Dataconsumer


class Trainer(Dataconsumer):
    def __init__(self, args):
        super().__init__(args)
        self.get_data("train", path=self.args.path)
        self.get_data("val", path=self.args.path)
        # Value is set to false as we want to fine-tune always the base model from Huggingface
        self.pretrained = False
        self.set_model()
        self.set_tokenizer()
        self.validate_tokenizer()

    def __tokenize_function(self, row):
        return self.tokenizer(
            row["comments"],
            padding='max_length',
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        )

    def train(self):
        column_names = self.train_data.column_names

        self.train_data = self.train_data.map(
            self.__tokenize_function,
            batched=True,
            remove_columns=column_names
        )

        self.val_data = self.val_data.map(
            self.__tokenize_function,
            batched=True,
            remove_columns=column_names
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,  # default
            mlm_probability=0.15,  # default
            return_tensors="pt",  # default
            # pad_to_multiple_of=5
        )

        steps_per_epoch = int(len(self.train_data) / self.args.batch_size)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            logging_dir=self.logging_dir,
            num_train_epochs=self.args.epochs,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            warmup_steps=self.args.warmup_steps,
            save_steps=steps_per_epoch,
            save_total_limit=1,
            weight_decay=self.args.weight_decay,
            learning_rate=self.args.learning_rate,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            greater_is_better=False,
            # eval_accumulation_steps=1
            # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/3
        )

        trainer = TransformerTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            tokenizer=self.tokenizer
        )

        trainer.train()
        trainer.save_model(self.path_save_model)

        self.delete_logs()
