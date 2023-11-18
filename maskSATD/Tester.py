import copy
import json
import math
from Dataconsumer import Dataconsumer
from utils import flat_list, \
    get_patterns


class Evaluation:

    def __init__(self):
        self.top_1 = 0
        self.top_3 = 0
        self.top_5 = 0
        self.frequency = {}

    def eval_accuracy(self, predictions, target, eval_function):
        self.top_1 += eval_function(predictions, target, 1)
        self.top_3 += eval_function(predictions, target, 3)
        self.top_5 += eval_function(predictions, target, 5)

    def __add_parameter_to_dict(self, dict, parameter, default):
        if parameter not in dict:
            dict[parameter] = default
        return dict

    def __increment_parameter(self, dict, parameter):
        for elem in parameter:
            dict = self.__add_parameter_to_dict(dict, elem, 0)
            dict[elem] += 1

        return dict

    def eval_frequency(self, predictions, target, eval_function):
        target, prediction = eval_function(predictions, target, 1)
        self.frequency = self.__add_parameter_to_dict(self.frequency, target, {})
        self.frequency[target] = self.__increment_parameter(self.frequency[target], prediction)

    def print_frequency(self):
        for mask, dict in self.frequency.items():
            print(mask)
            for key, value in dict.items():
                print('     ', key, ' : ', value)


class Tester(Dataconsumer):
    def __init__(self, args):
        super().__init__(args)
        self.get_data("test", return_format="List", path=self.args.path)
        self.set_model()
        self.set_tokenizer()
        self.validate_tokenizer()
        self.confusing = self.args.patterns == "Confusing"

    def __eval(self, predicted_token_ids_top_5, id, top_k):
        outputs = [self.tokenizer.convert_ids_to_tokens(id).replace("Ġ", "") for id in
                   predicted_token_ids_top_5[0:top_k]]

        if self.tokenizer.convert_ids_to_tokens(id).replace("Ġ", "") in outputs:
            return 1

        return 0

    def __eval_exact(self, predicted_token_ids_top_5, id, top_k):
        outputs = [self.tokenizer.convert_ids_to_tokens(id) for id in
                   predicted_token_ids_top_5[0:top_k]]
        if self.tokenizer.convert_ids_to_tokens(id) in outputs:
            return 1

        return 0

    def __eval_satd(self, predicted_token_ids_top_5, tokens, top_k):
        outputs = [self.tokenizer.convert_ids_to_tokens(id).replace("Ġ", " ") for id in
                   predicted_token_ids_top_5[0:top_k]]

        for token in tokens:
            if token in outputs:
                return 1

        return 0

    def __eval_frequency(self, predicted_tokens_ids_top_5, token, top_k):
        target = self.tokenizer.convert_ids_to_tokens(token).replace("Ġ", "").replace(" ", "")
        prediction = [self.tokenizer.convert_ids_to_tokens(id).replace("Ġ", "").replace(" ", "") for id in
                      predicted_tokens_ids_top_5[0:top_k]]

        return target, prediction

    def __eval_frequency_exact(self, predicted_tokens_ids_top_5, token, top_k):
        target = self.tokenizer.convert_ids_to_tokens(token)
        prediction = [self.tokenizer.convert_ids_to_tokens(id) for id in predicted_tokens_ids_top_5[0:top_k]]

        return target, prediction

    def test(self):
        tokens = [self.tokenizer.tokenize(comment) for comment in self.test_data]
        token_ids_input = [self.tokenizer.convert_tokens_to_ids(tokens_pre_input) for tokens_pre_input in tokens]
        vocab_ids = [self.tokenizer.convert_tokens_to_ids(token.replace(" ", "Ġ")) if self.tokenizer.convert_tokens_to_ids(token.replace(" ", "Ġ")) != 3 else self.tokenizer.convert_tokens_to_ids(token) for token in self.vocab]
        indices_to_mask = []

        for token_ids_per_sentence in token_ids_input:
            indices_to_mask_per_sentence = []
            # We want only to mask base tokens.
            # E.g. if we test MAT vs SATD, MAT would be the base
            for filter_vocab in self.vocab_base:
                tokens = self.tokenizer.tokenize(filter_vocab)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                indices_found = []
                for index in range(len(token_ids)):
                    token_id = token_ids[index]
                    if index == 0:
                        indices_found = [[i] for i, id in enumerate(token_ids_per_sentence) if token_id == id]
                    elif index > 0:
                        for i in range(len(indices_found)):
                            try:
                                check_next_index = indices_found[i][index - 1] + 1
                                if token_ids_per_sentence[check_next_index] == token_ids[index]:
                                    indices_found[i].append(check_next_index)
                            except:
                                # can be skipped as the sequence of tokens is not right
                                # e.g. t|od|o t was detected but not od so it was not added; however, it is removed later from the list
                                pass

                if len([indices for indices in indices_found if len(indices) == len(tokens)]) > 0:
                    indices_to_mask_per_sentence.append(
                        [indices for indices in indices_found if len(indices) == len(tokens)])
            indices_to_mask.append(indices_to_mask_per_sentence)

        for i in range(len(indices_to_mask)):
            indices_to_mask[i] = flat_list(flat_list(indices_to_mask[i]))

        total_predictions = 0
        predicted_full_vocab = Evaluation()
        predicted_reduced_vocab = Evaluation()
        predicted_full_vocab_exact = Evaluation()
        predicted_reduced_vocab_exact = Evaluation()
        if self.confusing:
            predicted_full_vocab_confusing = Evaluation()
            predicted_reduced_vocab_confusing = Evaluation()

        for index in range(len(indices_to_mask)):
            print(index, "/", len(indices_to_mask))
            token_ids_sentence_DC = copy.deepcopy(token_ids_input[index])
            # adds all the masks according to the indices for this input
            for index_tokens in range(len(indices_to_mask[index])):
                token_ids_sentence_DC[indices_to_mask[index][index_tokens]] = self.tokenizer.mask_token_id
            # print("Input sentence:\n", tokenizer.decode(token_ids_input[index]))
            # print("Sentence with all masked token:\n", tokenizer.decode(token_ids_sentence_DC))
            encoded_input = self.tokenizer(
                self.tokenizer.decode(token_ids_sentence_DC),
                return_tensors="pt",
                max_length=self.args.max_length,
                truncation='longest_first',
                padding='longest'
            )
            predictions = self.model(**encoded_input).logits
            mask_token_index = (encoded_input.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            for mask_index in range(len(mask_token_index)):
                total_predictions += 1
                # predicted_token_id = predictions[0, mask_token_index[mask_index]].argmax(axis=-1) # from example
                predicted_token_ids = predictions[0, mask_token_index[mask_index]].argsort(axis=-1, descending=True,
                                                                                           stable=False).numpy().tolist()
                predicted_token_ids_top_5 = predicted_token_ids[0:5]

                filtered_prediction_token_id_top_5 = [token_id for token_id in predicted_token_ids if
                                                      token_id in vocab_ids][0:5]
                # print("Top 5 prediction", [tokenizer.decode(id) for id in predicted_token_ids_top_5], predicted_token_ids_top_5)
                # print("Top 5 filtered prediction", [tokenizer.convert_ids_to_tokens(id) for id in filtered_prediction_token_id_top_5], filtered_prediction_token_id_top_5)

                if self.args.evaluation_strategy == "accuracy":
                    # full vocab
                    predicted_full_vocab.eval_accuracy(
                        predicted_token_ids_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval
                    )

                    # filtered
                    predicted_reduced_vocab.eval_accuracy(
                        filtered_prediction_token_id_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval
                    )

                    # full vocab exact
                    predicted_full_vocab_exact.eval_accuracy(
                        predicted_token_ids_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval_exact
                    )

                    # filtered exact
                    predicted_reduced_vocab_exact.eval_accuracy(
                        filtered_prediction_token_id_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval_exact
                    )

                    if self.confusing:
                        # full vocab
                        predicted_full_vocab_confusing.eval_accuracy(
                            predicted_token_ids_top_5,
                            get_patterns("SATD"),
                            self.__eval_satd
                        )

                        # filtered
                        predicted_reduced_vocab_confusing.eval_accuracy(
                            filtered_prediction_token_id_top_5,
                            get_patterns("SATD"),
                            self.__eval_satd
                        )

                elif self.args.evaluation_strategy == "frequency":
                    predicted_full_vocab.eval_frequency(
                        predicted_token_ids_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval_frequency
                    )

                    predicted_reduced_vocab.eval_frequency(
                        filtered_prediction_token_id_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval_frequency
                    )

                    predicted_full_vocab_exact.eval_frequency(
                        predicted_token_ids_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval_frequency_exact
                    )

                    predicted_reduced_vocab_exact.eval_frequency(
                        filtered_prediction_token_id_top_5,
                        token_ids_input[index][indices_to_mask[index][mask_index]],
                        self.__eval_frequency_exact
                    )

                # print("----")

        print("#################################################################")
        print("#{}{}{}#".format(" "*math.ceil((63-len(self.dataset))/2), self.dataset, " "*math.ceil((63-len(self.dataset))/2)))
        print("#################################################################")


        print("Run is completed, in the following the evaluation can be found: \n")
        if self.args.evaluation_strategy == "accuracy":
            print("Top 1 for full vocab", predicted_full_vocab.top_1 / total_predictions)
            print("Top 3 for full vocab", predicted_full_vocab.top_3 / total_predictions)
            print("Top 5 for full vocab", predicted_full_vocab.top_5 / total_predictions)
            print("Top 1 for reduced vocab", predicted_reduced_vocab.top_1 / total_predictions)
            print("Top 3 for reduced vocab", predicted_reduced_vocab.top_3 / total_predictions)
            print("Top 5 for reduced vocab", predicted_reduced_vocab.top_5 / total_predictions)
            print("Top 1 for full vocab exact", predicted_full_vocab_exact.top_1 / total_predictions)
            print("Top 3 for full vocab exact", predicted_full_vocab_exact.top_3 / total_predictions)
            print("Top 5 for full vocab exact", predicted_full_vocab_exact.top_5 / total_predictions)
            print("Top 1 for reduced vocab exact", predicted_reduced_vocab_exact.top_1 / total_predictions)
            print("Top 3 for reduced vocab exact", predicted_reduced_vocab_exact.top_3 / total_predictions)
            print("Top 5 for reduced vocab exact", predicted_reduced_vocab_exact.top_5 / total_predictions)
            if self.confusing:
                print("\n-----------------------------------------\n")
                print("Top 1 for full vocab", predicted_full_vocab_confusing.top_1 / total_predictions)
                print("Top 3 for full vocab", predicted_full_vocab_confusing.top_3 / total_predictions)
                print("Top 5 for full vocab", predicted_full_vocab_confusing.top_5 / total_predictions)
                print("Top 1 for reduced vocab", predicted_reduced_vocab_confusing.top_1 / total_predictions)
                print("Top 3 for reduced vocab", predicted_reduced_vocab_confusing.top_3 / total_predictions)
                print("Top 5 for reduced vocab", predicted_reduced_vocab_confusing.top_5 / total_predictions)
            print("There where a total number of {} masks predicted".format(total_predictions))
        elif self.args.evaluation_strategy == "frequency":
            result = {"full_vocab": predicted_full_vocab.frequency,
                      "full_vocab_exact": predicted_full_vocab_exact.frequency,
                      "reduced_vocab": predicted_reduced_vocab.frequency,
                      "reduced_vocab_exact": predicted_reduced_vocab_exact.frequency}
            print("Prediction with full vocabulary")
            print(predicted_full_vocab.print_frequency())
            print("Prediction with full vocabulary (exact)")
            print(predicted_full_vocab_exact.print_frequency())
            print("Prediction with reduced vocabulary")
            print(predicted_reduced_vocab.print_frequency())
            print("Prediction with reduced vocabulary (exact)")
            print(predicted_reduced_vocab_exact.print_frequency())

            # Overwrites an existing file
            with open("{}.json".format(self.path_save_model), "w") as outfile:
                json.dump(result, outfile)

        self.delete_logs()
        self.delete_model()
