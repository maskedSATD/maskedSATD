import pandas as pd
import re
from argparse import ArgumentParser
from datasets import Dataset
from tqdm import tqdm


def get_mat():
    return ["xxx", "todo", "hack", "fixme"]


def get_satd():
    return ["hack", "retarded", "at a loss", "stupid", "remove this code", "ugly", "take care",
            "something's gone wrong", "nuke", "is problematic", "may cause problem", "hacky",
            "unknown why we ever experience this", "treat this as a soft error", "silly", "workaround for bug",
            "kludge", "fixme", "this isn't quite right", "trial and error", "give up", "this is wrong",
            "hang our heads in shame", "temporary solution", "causes issue", "something bad is going on",
            "cause for issue", "this doesn't look right", "is this next line safe",
            "this indicates a more fundamental problem", "temporary crutch", "this can be a mess",
            "this isn't very solid", "this is temporary and will go away", "is this line really safe",
            "there is a problem", "some fatal error", "something serious is wrong", "don't use this", "get rid of this",
            "doubt that this would work", "this is bs", "give up and go away", "risk of this blowing up",
            "just abandon it", "prolly a bug", "probably a bug", "hope everything will work", "toss it", "barf",
            "something bad happened", "fix this crap", "yuck", "certainly buggy", "remove me before production",
            "you can be unhappy now", "this is uncool", "bail out", "it doesn't work yet", "crap", "inconsistency",
            "abandon all hope", "kaboom"]


# The confusing pattern have been selected based on the Fleiss Kappa.
# For more details look into the folder fleiss_kappa, which contains the raw data and the evaluation script
def get_confusion_terms():
    return ["why", "fix", "need to", "this needs", "work around", "temporary", "should this", "should we",
            "do we need", "perhaps not really necessary", "should be created", "not absolutely necessary",
            "workaround for", "why not", "remove", "element should be created", "bug", "not sure",
            "nasty hardcoded", "need this", "not used", "should really", "issue", "critic", "later", "revisit",
            "problem", "bad", "wrong", "problems occur", "remove this", "wtf", "not yet handled", "should not be",
            "this perhaps not really necessary", "quick fix", "implement this", "workaround", "null", "to handle",
            "should probably be", "error", "delete", "would be better", "this needs to be", "missing", "be removed",
            "empty block", "move this to", "fix this", "is it the best way", "deal with", "to check", "errors",
            "smarter", "broken", "gross", "do n't know", "how to handle", "resolve", "would be nice"]


def get_bad_functions():
    return ["strcpy", "strcpyA", "strcpyW", "wcscpy", "_tcscpy", "_mbscpy", "StrCpy", "StrCpyA",
            "StrCpyW", "lstrcpy", "lstrcpyA", "lstrcpyW", "_tccpy", "_mbccpy", "_ftcscpy", "strcpyA",
            "strcpyW", "wcscpy", "_tcscpy", "_mbscpy", "StrCpy", "StrCpyA", "StrCpyW", "lstrcpy", "lstrcpyA",
            "lstrcpyW", "_tccpy", "_mbccpy", "_ftcscpy", "strcat", "strcatA", "strcatW", "wcscat", "_tcscat",
            "_mbscat", "StrCat", "StrCatA", "StrCatW", "lstrcat", "lstrcatA", "lstrcatW", "StrCatBuff",
            "StrCatBuffA", "StrCatBuffW", "StrCatChainW", "_tccat", "_mbccat", "_ftcscat", "sprintfW", "sprintfA",
            "wsprintf", "wsprintfW", "wsprintfA", "sprintf", "swprintf", "_stprintf", "wvsprintf", "wvsprintfA",
            "wvsprintfW", "vsprintf", "_vstprintf", "vswprintf", "strncpy", "wcsncpy", "_tcsncpy", "_mbsncpy",
            "_mbsnbcpy", "StrCpyN", "StrCpyNA", "StrCpyNW", "StrNCpy", "strcpynA", "StrNCpyA", "StrNCpyW", "lstrcpyn",
            "lstrcpynA", "lstrcpynW", "strncat", "wcsncat", "_tcsncat", "_mbsncat", "_mbsnbcat", "StrCatN", "StrCatNA",
            "StrCatNW", "StrNCat", "StrNCatA", "StrNCatW", "lstrncat", "lstrcatnA", "lstrcatnW", "lstrcatn", "gets",
            "_getts", "_gettws", "IsBadWritePtr", "IsBadHugeWritePtr", "IsBadReadPtr", "IsBadHugeReadPtr",
            "IsBadCodePtr", "IsBadStringPtr", "memcpy", "RtlCopyMemory", "CopyMemory", "wmemcpy", "lstrlen",
            "wnsprintf", "wnsprintfA", "wnsprintfW", "_snwprintf", "_snprintf", "_sntprintf", "_vsnprintf",
            "vsnprintf", "_vsnwprintf", "_vsntprintf", "wvnsprintf", "wvnsprintfA", "wvnsprintfW", "strtok", "_tcstok",
            "wcstok", "_mbstok", "makepath", "_tmakepath", "_makepath", "_wmakepath", "_splitpath", "_tsplitpath",
            "_wsplitpath", "scanf", "wscanf", "_tscanf", "sscanf", "swscanf", "_stscanf", "snscanf", "snwscanf",
            "_sntscanf", "_itoa", "_itow", "_i64toa", "_i64tow", "_ui64toa", "_ui64tot", "_ui64tow", "_ultoa", "_ultot",
            "_ultow", "CharToOem", "CharToOemA", "CharToOemW", "OemToChar", "OemToCharA", "OemToCharW",
            "CharToOemBuffA", "CharToOemBuffW", "alloca", "_alloca", "ChangeWindowMessageFilter"]


def mutate_badFunctions(patterns):
    return map(lambda x: "([^a-zA-Z0-9]|^){}\(".format(x), patterns)


# taken from https://github.com/melegati/vulsatd-dataset/blob/master/mat.py
# which replicated it from https://github.com/Naplues/MAT/blob/master/src/main/methods/Mat.java
def has_task_words(x, patterns=get_mat(), index=0):
    regex = re.compile('[^a-zA-Z\s]')
    regex_space = re.compile('[\s]')

    x_mod = regex.sub("", x)  # replaces non-alpha characters
    x_mod = regex_space.sub(" ", x_mod)  # converts tab and other space characters into a single space

    words = x_mod \
        .lower() \
        .replace("'", "") \
        .split(' ')

    words = [word for word in words if 20 > len(word) > 2]  # removes words which are too short or too long

    for word in words:
        for key in patterns:
            if word.startswith(key) or word.endswith(key):
                if 'xxx' in word and word != 'xxx':
                    return False
                else:
                    return True
    return False

def has_task_words_values(x, patterns=get_mat(), index=0):
    regex = re.compile('[^a-zA-Z\s]')
    regex_space = re.compile('[\s]')

    x_mod = regex.sub("", x)  # replaces non-alpha characters
    x_mod = regex_space.sub(" ", x_mod)  # converts tab and other space characters into a single space

    words = x_mod \
        .lower() \
        .replace("'", "") \
        .split(' ')

    words = [word for word in words if 20 > len(word) > 2]  # removes words which are too short or too long
    result = []
    for word in words:
        for key in patterns:
            if word.startswith(key) or word.endswith(key):
                if 'xxx' in word and word != 'xxx':
                    pass
                else:
                    result.append(key)
    return result


def predict_mat(target, patterns=get_mat()):
    return [has_task_words(x, patterns) for x in target]


def flat_list(list):
    return [item for sublist in list for item in sublist]


def load_data(path, nrows=None):
    df = pd.concat(
        [chunk for chunk in tqdm(pd.read_csv(path, chunksize=1000, lineterminator="\n", nrows=nrows), desc="Load csv...")])
    return df


def load_split_dataset(data_folder):
    train = load_data(data_folder + "/train.csv")
    train.fillna('', inplace=True)
    # train = train[0:5]  # TODO: remove
    # train['Class'] = train.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)

    val = load_data(data_folder + "/val.csv")
    val.fillna('', inplace=True)
    # val = val[0:5]  # TODO: remove
    # val['Class'] = val.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)

    test = load_data(data_folder + "/test.csv")
    test.fillna('', inplace=True)
    # test = test[0:5]  # TODO: remove
    # test['Class'] = test.apply(lambda row: row.Vulnerable * 2 + row.SATD, axis=1)

    return train, val, test


def remove_non_SATD(data, patterns):
    if patterns == 'MAT':
        return data[data['MAT'] == 1]

    if patterns == "BadFunctions":
        return data[(data["MAT"] == 1) | (data["CWE_242_676"] == 1)]

    return data[(data["MAT"] == 1) | (data[patterns] == 1)]


def remove_function_without_patterns(data, patterns):
    pattern = "|".join(remove_spaces_patterns(patterns)).replace("xxx", "\s?xxx\s?")
    regex = re.compile(pattern, re.IGNORECASE)
    result = []
    for comment in data:
        if len(re.findall(regex, comment)) > 0:
            result.append(comment)
    return result


def extract_comments(data, pl):
    comment_regex = get_comment_regex(pl)
    result = []
    for index in range(len(data)):
        if "Code" in data.keys():
            detected_comments = re.findall(comment_regex, data["Code"][data.index[index]])
            comments = ' '.join([comment[0] for comment in detected_comments])
            result.append(comments)
        else:
            if "commenttext" in data.keys():  # MD dataset
                result.append(data["commenttext"][data.index[index]])
            else:
                print("Keys are not supported, expected to include 'Code' or 'commenttext as a key!'")
                print(data.keys())
                quit()
    return result


def get_patterns(pattern_type):
    if pattern_type == "MAT":
        return get_mat()
    elif pattern_type == "SATDWithoutMAT":
        return get_satd()
    elif pattern_type == "SATD-PS":
        patterns = get_satd()
        patterns.append("todo")
        patterns.append("xxx")
        return patterns
    elif pattern_type == "Confusion":
        patterns = get_confusion_terms() + get_satd()
        patterns.append("xxx")
        patterns.append("todo")
        return patterns
    elif pattern_type == "BadFunctions":
        patterns = get_bad_functions() + get_mat()
        return patterns

    print("The requested set of pattern ({}) does not exist! Exiting...".format(pattern_type))
    quit()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if "{}".format(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif "{}".format(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        parser = ArgumentParser()
        raise parser.error('Boolean value expected.')


# Removes spaces with changing the existing capitalization of the token
# e.g. to do -> todo
# e.g. TO DO -> todo
# e.g. To Do -> todo
def replace_spaces_with_nothing(string, pattern):
    array = [re.sub(" ", "", re.sub("\W", "", pattern)) for pattern in
             re.findall('\W?' + pattern + '\W?', string, re.IGNORECASE)]
    for replace in array:
        string = re.sub(pattern, replace.lower(), string, count=1, flags=re.IGNORECASE)

    return string


def remove_spaces_from_patterns(data, patterns=[]):
    result = []
    for comment in data:
        for pattern in patterns:
            comment = replace_spaces_with_nothing(comment, pattern)

        comment = replace_spaces_with_nothing(comment, "to do")
        comment = replace_spaces_with_nothing(comment, "fix me")

        result.append(comment)

    return result


def remove_spaces_patterns(patterns):
    return [re.sub("\s", "", pattern) for pattern in patterns]


def extract_comments_containing_patterns(data, patterns, pl):
    if patterns == None:
        print("patterns not passed... Exiting...")
        quit()
    data = remove_non_SATD(data, patterns)
    data = extract_comments(data, pl)
    data = remove_spaces_from_patterns(data, patterns=get_patterns(patterns))
    data = remove_function_without_patterns(data, patterns=get_patterns(patterns))

    return data


def extract_functions_containing_patterns(data, patterns):
    # if the column is present in the data the time-consuming task of labelling the data can be skipped.
    if "CWE_242_676" in data.columns:
        return data[data['CWE_242_676'] == True]["Code"]

    data = [data["Code"][data.index[index]] for index in range(len(data))]
    result = []
    for code in data:
        matches = re.findall("|".join(patterns), code, re.IGNORECASE)
        if len(matches) > 0:
            result.append(code)

    return result


def convert_data(data, return_format):
    if len(data) > 0:
        data = flat_list(data)
        # data = data[0:5] # TODO remove
        if return_format == "Dataset":
            df = pd.DataFrame(data=data)
            if df.shape[1] != 1:
                df = df.T
            df.columns = ["comments"]
            return Dataset.from_pandas(df)

        return data

    print("There is not data to be analysed. Exiting...")
    quit()


def get_comment_regex(pl):
    comment_regex = None
    pl = pl.lower()
    if pl == "go" or pl == "java" or pl == "javascript" or pl == "c":
        comment_regex = re.compile('((\/\*([\s\S]*?)\*\/)|((?<!:)\/\/.*))', re.MULTILINE)
    elif pl == "php":
        comment_regex = re.compile('(((\/\*([\s\S]*?)\*\/)|((?<!:)\/\/.*))|(#.+?(?=\?\>))|(#.*))', re.MULTILINE)
    elif pl == "ruby":
        comment_regex = re.compile('((#.*)|(\=begin[\s\S]*?\=end))', re.MULTILINE)
    elif pl == "python":
        comment_regex = re.compile('((#.*)|(\'|"){3}[\s\S]*?(\'|"){3})', re.MULTILINE)
    else:
        print("Programming language is not covered... Exiting...")
        quit()

    return comment_regex


def load_comments_containing_pattern(path, train=True, val=True, test=True, return_format="List", patterns="",
                                     comments_only=True):

    print(path)
    train_mod, val_mod, test_mod = load_split_dataset(path)
    data = []
    pl = path.split("/")[-1] if "csn" in path else "c"
    if train:
        data.append(extract_comments_containing_patterns(train_mod, patterns=patterns, pl=pl) if comments_only else
                    extract_functions_containing_patterns(train_mod, patterns=patterns))
    if val:
        data.append(extract_comments_containing_patterns(val_mod, patterns=patterns, pl=pl) if comments_only else
                    extract_functions_containing_patterns(val_mod, patterns=patterns))
    if test:
        data.append(extract_comments_containing_patterns(test_mod, patterns=patterns, pl=pl) if comments_only else
                    extract_functions_containing_patterns(test_mod, patterns=patterns))

    return convert_data(data, return_format)


def get_full_dataset(train=False, val=False, test=False, return_format="List", patterns="",
                     comments_only=True, path="", dataset="all"):
    data = []

    folders = []

    if dataset == "all" or dataset == "Devign":
        folders.append("devign")
    if dataset == "all" or dataset == "CSN":
        folders.append("csn/go")
        folders.append("csn/java")
        folders.append("csn/javascript")
        folders.append("csn/php")
        folders.append("csn/python")
        folders.append("csn/ruby")
    if dataset == "all" or dataset == "WeakSATD":
        folders.append("cfl")
    if dataset == "all" or dataset == "Big-Vul":
        folders.append("big-vul-10")
    if dataset == "all" or dataset == "MD":
        folders.append("md")

    for folder in folders:
        print("Loading data from {}...".format(folder))
        data.append(
            load_comments_containing_pattern("{}/{}".format(path, folder), train=train, val=val,
                                             test=test, return_format="List", patterns=patterns,
                                             comments_only=comments_only))

    return convert_data(data, return_format)


def generate_full_vocab(filtered_vocab, add_space=True, add_lower=True, break_input=False):
    vocab_len = len(filtered_vocab)
    if break_input:
        filtered_vocab = [vocab.split(" ") for vocab in filtered_vocab]
        filtered_vocab = flat_list(filtered_vocab)

    if add_space:
        for index in range(vocab_len):
            filtered_vocab.append(" {}".format(filtered_vocab[index]))

    vocab_len = len(filtered_vocab)
    if add_lower:
        for index in range(vocab_len):
            filtered_vocab.append(filtered_vocab[index].lower())

    return filtered_vocab


def read_args_single_input():
    parser = ArgumentParser()

    parser.add_argument('--train', type=bool, required=False, default=False)
    parser.add_argument('--val', type=bool, required=False, default=False)
    parser.add_argument('--test', type=bool, required=False, default=False)

    return parser.parse_args()


def create_output_path(start_path, args):
    return "{}_{}_{}_{}_{}_{}_vs_{}_for_{}".format(
        start_path,
        args.learning_rate,
        args.epochs,
        args.model.replace("/", "-"),
        args.batch_size,
        args.base_patterns,
        args.patterns,
        args.dataset
    )
