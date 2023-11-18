from utils import get_full_dataset, \
    create_output_path, \
    get_patterns, \
    remove_spaces_patterns, \
    generate_full_vocab
import shutil
import os
from transformers import RobertaTokenizer, \
    RobertaForMaskedLM, \
    BertTokenizer, \
    BertForMaskedLM

def find_vocab(name, initial_vocab, tokenizer, tokenizer_name, add_space, add_lower, break_input, output=True):
    vocab = generate_full_vocab(initial_vocab, add_space=add_space, add_lower=add_lower, break_input=break_input)
    vocab_tokenized = [tokenizer.tokenize(vocab) for vocab in vocab]
    # vocab_ids = [[tokenizer.convert_tokens_to_ids(token) for token in tokens] for tokens in vocab_tokenized]
    tokens_to_be_encoded = list(set([vocab[index] for index in range(len(vocab)) if len(vocab_tokenized[index]) > 1]))
    tokens_are_encoded = list(set([vocab[index] for index in range(len(vocab)) if len(vocab_tokenized[index]) == 1]))
    if output:
        # print(vocab)
        # print(vocab_tokenized)
        # print(vocab_ids)
        print(
            "\nThe following evaluation was performed for the {} keywords.\nThe input was {} split at the space character if it was present".format(
                name, "" if break_input else "not"))
        print("The following tokens are not encoded in the base vocabulary of {}: (there are {} tokens)\n{}".format(
            tokenizer_name, len(tokens_to_be_encoded), tokens_to_be_encoded))
        print("The following tokens are encoded in the base vocabulary of {}: (there are {} tokens)\n{}".format(
            tokenizer_name, len(tokens_are_encoded), tokens_are_encoded))

    return len(tokens_to_be_encoded), len(tokens_are_encoded), tokens_to_be_encoded, tokens_are_encoded


class Dataconsumer(object):
    def __init__(self, args):
        self.vocab_base = None
        self.vocab = None
        self.model = None
        self.tokenizer = None
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.args = args
        self.path_save_model = create_output_path("{}/model".format(self.args.save_path_prefix), self.args)
        self.output_dir = create_output_path("{}/bert-news".format(self.args.save_path_prefix), self.args)
        self.logging_dir = create_output_path("{}/LMlogs".format(self.args.save_path_prefix), self.args)
        self.patterns = get_patterns(self.args.patterns)
        self.base_patterns = get_patterns(self.args.base_patterns)
        self.pretrained = self.args.pretrained
        self.dataset = self.args.dataset

    def set_model(self):
        if os.path.isdir(self.path_save_model) and self.pretrained:
            print("Our model")
            if self.args.model == "bert-base-uncased":
                self.model = BertForMaskedLM.from_pretraiend(self.path_save_model)
            else:
                self.model = RobertaForMaskedLM.from_pretrained(self.path_save_model)
        else:
            print("Model from Huggingface")
            if self.args.model == "bert-base-uncased":
                self.model = RobertaForMaskedLM.from_pretrained(self.args.model)
            else:
                self.model = RobertaForMaskedLM.from_pretrained(self.args.model)

    def set_tokenizer(self):
        if os.path.isdir(self.path_save_model) and self.pretrained:
            print("Our model")
            if self.args.model == "bert-base-uncased":
                self.tokenizer = BertTokenizer.from_pretrained(self.path_save_model)
            else:
                self.tokenizer = RobertaTokenizer.from_pretrained(self.path_save_model)
        else:
            print("Model from Huggingface")
            if self.args.model == "bert-base-uncased":
                self.tokenizer = BertTokenizer.from_pretrained(self.args.model)
            else:
                self.tokenizer = RobertaTokenizer.from_pretrained(self.args.model)

    def get_data(self, data_type, return_format="Dataset", path=""):
        if data_type == "train":
            self.train_data = get_full_dataset(train=True, val=False, test=False, return_format=return_format,
                                               patterns=self.args.patterns,  # TODO remove it after the experiment
                                               comments_only=self.args.comments_only,
                                               path=path,
                                               dataset=self.dataset)
        elif data_type == "val":
            self.val_data = get_full_dataset(train=False, val=True, test=False, return_format=return_format,
                                             patterns=self.args.patterns,  # TODO remove it after the experiment
                                             comments_only=self.args.comments_only,
                                             path=path,
                                             dataset=self.dataset)
        elif data_type == "test":
            self.test_data = get_full_dataset(train=False, val=False, test=True, return_format=return_format,
                                              patterns=self.args.base_patterns,
                                              comments_only=self.args.comments_only,
                                              path=path,
                                              dataset=self.dataset)

    def validate_tokenizer(self):
        print("Size of the training set: ", len(self.train_data))
        print("Size of the validation set: ", len(self.val_data))
        print("Size of the test set: ", len(self.test_data))

        self.patterns = remove_spaces_patterns(self.patterns)
        results = find_vocab(self.patterns, self.patterns, self.tokenizer, self.args.model, add_space=True,
                             add_lower=False,
                             break_input=False,
                             output=False)
        results_base = find_vocab(self.base_patterns, self.base_patterns, self.tokenizer, self.args.model,
                                  add_space=True,
                                  add_lower=False,
                                  break_input=False,
                                  output=False)
        self.vocab = results[2] + results[3]
        self.vocab_base = results_base[2] + results_base[3]
        tokens_to_be_added = results[2]

        print("Tokens not included in the model: \n", tokens_to_be_added)
        print("All tokens of interest: \n", self.vocab)
        print(
            "The tokens are not within the vocabulary or the model, such that they are encoded with the id 3, "
            "which is the is for unknown\n",
            self.tokenizer.convert_tokens_to_ids(tokens_to_be_added))

        tokens_added = self.tokenizer.add_tokens(tokens_to_be_added)

        print("Tokens added to the tokenizer (added - total number of tokens)")
        print(tokens_added, len(self.vocab))
        if tokens_added > 0:

            # resize the embeddings matrix of the model
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(
                "After we have added the tokens to the vocabulary, we receive the following encoding.\nIt should be "
                "noticed that the ids are in sequence as the new tokens are all added at the end of the vocabulary\n",
                self.tokenizer.convert_tokens_to_ids(tokens_to_be_added))
            print("Now we can also convert the ids back to the original tokens\n",
                  [self.tokenizer.decode([id]) for id in self.tokenizer.convert_tokens_to_ids(tokens_to_be_added)])
        else:
            print("The model contains already all the tokens, such that we do not need to add them anymore!")

        print(self.vocab)
        print(self.vocab_base)

    def delete_logs(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.rmtree(self.logging_dir, ignore_errors=True)

        print("The logs have been successfully deleted!")

    def delete_model(self):
        if self.args.delete_model:
            shutil.rmtree(self.path_save_model, ignore_errors=True)
            print("The model has been successfully deleted!")
