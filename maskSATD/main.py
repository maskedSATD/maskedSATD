from argparse import ArgumentParser
from utils import str2bool
from Trainer import Trainer
from Tester import Tester


def read_args():
    parser = ArgumentParser()
    # https://huggingface.co/roberta-base
    # https://huggingface.co/microsoft/codebert-base
    # https://huggingface.co/huggingface/CodeBERTa-small-v1
    # https://huggingface.co/microsoft/unixcoder-base
    # https://huggingface.co/neulab/codebert-c
    # https://huggingface.co/bert-base-uncased
    parser.add_argument('--model', choices=['roberta-base',
                                            'microsoft/codebert-base-mlm',
                                            'huggingface/CodeBERTa-small-v1',
                                            'microsoft/unixcoder-base',
                                            "neulab/codebert-c",
                                            # 'xlm-roberta-base',  # multilingual roberta with 88 languages, the dataset is genereated from text taken from the internet (no further specification) and wikipedia
                                            # 'bert-base-uncased'  # currently not working
                                            ],
                        default="microsoft/codebert-base-mlm",
                        help="The following models are available: roberta-base, microsoft/codebert-base-mlm, "
                             "huggingface/CodeBERTa-small-v1, microsoft/unixcoder-base")
    parser.add_argument('--pretrained', type=str2bool, default=False, help='With "True" the tester will load the '
                                                                           'pretrained model, whereas for "False" it '
                                                                           'will take the model from Huggingface. The '
                                                                           'training process will use always the base '
                                                                           'model from Huggingface')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--learning-rate', default=5e-5, type=float)
    parser.add_argument('--max-length', default=512, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--comments-only', default=True, type=str2bool, help='With "True" only comments will be '
                                                                             'passed in the model, whereas for '
                                                                             '"False" the entire code block will be '
                                                                             'passed in the model')
    parser.add_argument('--test', default=False, type=str2bool)
    parser.add_argument('--train', default=False, type=str2bool)
    parser.add_argument('--delete-model', default=False, type=str2bool, help="Take care! This will delete the "
                                                                             "fine-tuned model at the end of the "
                                                                             "program!"
                                                                             "It is supposed to be used only during "
                                                                             "testing to save memory on the machine "
                                                                             "it is executed.")
    parser.add_argument('--patterns', choices=["MAT", "SATDWithoutMAT", "SATD-PS", "Confusion", "BadFunctions"], type=str, default="MAT")
    parser.add_argument('--base-patterns', choices=["MAT", "BadFunctions"], default="MAT")
    parser.add_argument('--path', type=str, required=True, help="Path to the folder containing the data which is used "
                                                                "for the training, validation and testing")
    parser.add_argument('--evaluation-strategy', type=str, choices=["accuracy", "frequency"], default="accuracy",
                        help="Decision if the experiment returns the accuracy of the prediction or the frequency of "
                             "the tokens the base-line was replaced with")
    parser.add_argument('--save-path-prefix', type=str, default=".")
    parser.add_argument('--dataset', type=str, choices=["all", "Devign", "CSN", "WeakSATD", "Big-Vul", "MD"], default="all")

    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    print(args)
    if args.train:
        trainer = Trainer(args)
        trainer.train()

    if args.test:
        tester = Tester(args)
        tester.test()
