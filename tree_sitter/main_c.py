from tree_sitter import Language, Parser
import pandas as pd
from tqdm import tqdm
from c_parser import CParser
import re
Language.build_library(
  # Store the library in the `build` directory
  'build/parser_c.so',

  # Include one or more languages
  [
    'vendor/tree-sitter-c',
  ]
)

PROJECTS = [
    "cfl",
    "devign",
    "big-vul-10"
]


INPUT_PATH = "/Users/moritzmock/PycharmProjects/masked-satd/eval/extract_samples/sample_CWE478.csv"
OUTPUT_PATH = "/Users/moritzmock/PycharmProjects/masked-satd/eval/extract_samples/sample_CWE478_CT.csv"

def load_data(path):
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(path, chunksize=1000), desc='Loading data')])
    return df

if __name__ == '__main__':
    LANGUAGE = Language("build/parser_c.so", "c")

    parser = Parser()
    parser.set_language(LANGUAGE)

    df = load_data(INPUT_PATH)
    df["code_tokens"] = str
    df["comment4code"] = "FIXME, this is a test"
    df["comment4code_short"] = "FIXME, this is a test"
    with tqdm(total=len(df.index), desc="Mapping data") as pbar:
        for idx in range(len(df.index)):
            pbar.update(1)
            code = df["Code"][df.index[idx]]
            tree = parser.parse(code.encode())
            code_tokens = CParser.get_definition(tree, code)
            df.loc[idx, "code_tokens"] = code_tokens.__str__()


    df.to_csv(OUTPUT_PATH, index=False)