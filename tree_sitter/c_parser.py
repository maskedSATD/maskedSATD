from typing import List, Dict, Any

from language_parser import LanguageParser


class CParser(LanguageParser):

    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        definitions = []
        CParser.traverse_node(tree.root_node.children, definitions)
        return definitions

    @staticmethod
    def traverse_node(node, result):
        if not isinstance(node, list):
            if len(node.children) == 0:
                result.append(node.text.decode())
                return
            else:
                node = node.children

        for sub_node in node:
            CParser.traverse_node(sub_node, result)

