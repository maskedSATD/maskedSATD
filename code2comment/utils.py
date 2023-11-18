import re
from argparse import ArgumentParser


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


def get_mat():
    return ["hack", "xxx", "fixme", "todo"]


def check_comments(comments, pattern):
    for comment in comments:
        if len(re.findall("|".join(pattern), comment, re.IGNORECASE)):
            return comment


def extract_comments(code, pl):
    comment_regex = get_comment_regex(pl)
    detected_comments = re.findall(comment_regex, code)
    return [comment[0] for comment in detected_comments]






class EvaluationAndTracker:
    def __init__(self, do_remove_non_ascii_symbols, do_remove_short_comments):
        self.counter_ascii_symbols = 0
        self.counter_short_comments = 0
        self.do_remove_non_ascii_symbols = do_remove_non_ascii_symbols
        self.do_remove_short_comments = do_remove_short_comments

    def __determiner(self, value):
        return False if value is not None else True

    def __check_for_non_ascii_symbols(self, input_string):
        """
        True: does contain non-ascii symbols
        False: does not contain non-ascii symbols
        """
        regex = re.compile('[^\x00-\x7F]')
        for letter in input_string:
            match = regex.match(letter)

            if not self.__determiner(match):
                return True

        return False

    def __check_for_too_short_comments(self, input_string):
        """
        True: comment is too short
        False: comment is not too short
        """
        cleaned_string = re.sub('[^\w]', "", input_string.lower())  # only keeps \w
        cleaned_string = re.sub("|".join(get_mat()), "", cleaned_string, re.IGNORECASE)  # removes MAT
        #return True if len(cleaned_string) <= 3 or len(cleaned_string) >= 50 else False  # assess the length of the comment
        return len(cleaned_string) <= 3

    def __ascii_check(self, target):
        if self.do_remove_non_ascii_symbols and self.__check_for_non_ascii_symbols(target):
            self.counter_ascii_symbols += 1
            return False

        return True

    def __length_checker(self, target):
        if self.do_remove_short_comments and self.__check_for_too_short_comments(target):
            self.counter_short_comments += 1
            return False

        return True

    def add_example_conditions(self, target):
        return self.__ascii_check(target) and self.__length_checker(target)

    def print(self):
        if self.do_remove_non_ascii_symbols:
            print(self.counter_ascii_symbols, "comments have been removed from the samples as they have contained "
                                              "non-ascii characters! E.g. the comment was written in chinese")

        if self.do_remove_short_comments:
            print(self.counter_short_comments, "comments have been removed from the samples as they are to short "
                                               "and did not provide any other information! E.g. the comment was "
                                               "only //TODO")
