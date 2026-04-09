from typing import TYPE_CHECKING, Optional

from astroid import nodes
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


class KeywordArgsChecker(BaseChecker):

    name = "keyword-args"
    msgs = {
        "W0001": (
            "Positional argument(s) found in call to function '%s'. Argument(s): %s",
            "non-keyword-args",
            "When ambiguity exists (ie more than one argument), "
            "using positional arguments can lead to errors.",
        ),
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)
        self.in_file_function = []

    def file_has_to_be_checked(self, filename: str) -> bool:
        return "test" not in filename.lower()

    def visit_functiondef(self, node: nodes.FunctionDef) -> None:
        self.in_file_function.append(node.name)

    def visit_call(self, node: nodes.Call) -> None:
        filename = self.linter.current_file
        if not self.file_has_to_be_checked(filename):
            return
        func_name = node.func.as_string()
        if self.is_in_file_function(func_name=func_name):
            positional_args = node.args
            if len(positional_args) > 1:
                arg_names = [arg.as_string() for arg in positional_args]
                self.add_message(
                    "non-keyword-args",
                    node=node,
                    args=(func_name, ", ".join(arg_names)),
                )

    def is_in_file_function(self, func_name: str) -> bool:
        return func_name in self.in_file_function


def register(linter: "PyLinter") -> None:
    linter.register_checker(checker=KeywordArgsChecker(linter))
