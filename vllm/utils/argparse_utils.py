# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Argument parsing utilities for vLLM."""

import json
import sys
import textwrap
from argparse import (
    Action,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
    Namespace,
    RawDescriptionHelpFormatter,
    _ArgumentGroup,
)
from collections import defaultdict
from typing import Any

import regex as re
import yaml

from vllm.logger import init_logger

logger = init_logger(__name__)


class SortedHelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def _split_lines(self, text, width):
        """
        1. Sentences split across lines have their single newlines removed.
        2. Paragraphs and explicit newlines are split into separate lines.
        3. Each line is wrapped to the specified width (width of terminal).
        """
        # The patterns also include whitespace after the newline
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(" ", text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()
    _json_tip: str = (
        "When passing JSON CLI arguments, the following sets of arguments "
        "are equivalent:\n"
        '   --json-arg \'{"key1": "value1", "key2": {"key3": "value2"}}\'\n'
        "   --json-arg.key1 value1 --json-arg.key2.key3 value2\n\n"
        "Additionally, list elements can be passed individually using +:\n"
        '   --json-arg \'{"key4": ["value3", "value4", "value5"]}\'\n'
        "   --json-arg.key4+ value3 --json-arg.key4+='value4,value5'\n\n"
    )
    _search_keyword: str | None = None

    def __init__(self, *args, **kwargs):
        # Set the default "formatter_class" to SortedHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        # Pop kwarg "add_json_tip" to control whether to add the JSON tip
        self.add_json_tip = kwargs.pop("add_json_tip", True)
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        # Enable the deprecated kwarg for Python 3.12 and below

        def parse_known_args(self, args=None, namespace=None):
            namespace, args = super().parse_known_args(args, namespace)
            for action in FlexibleArgumentParser._deprecated:
                if (
                    hasattr(namespace, dest := action.dest)
                    and getattr(namespace, dest) != action.default
                ):
                    logger.warning_once("argument '%s' is deprecated", dest)
            return namespace, args

        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                FlexibleArgumentParser._deprecated.add(action)
            return action

        class _FlexibleArgumentGroup(_ArgumentGroup):
            def add_argument(self, *args, **kwargs):
                deprecated = kwargs.pop("deprecated", False)
                action = super().add_argument(*args, **kwargs)
                if deprecated:
                    FlexibleArgumentParser._deprecated.add(action)
                return action

        def add_argument_group(self, *args, **kwargs):
            group = self._FlexibleArgumentGroup(self, *args, **kwargs)
            self._action_groups.append(group)
            return group

    def format_help(self):
        # Only use custom help formatting for bottom level parsers
        if self._subparsers is not None:
            return super().format_help()

        formatter = self._get_formatter()

        # Handle keyword search of the args
        if (search_keyword := self._search_keyword) is not None:
            # Normalise the search keyword
            search_keyword = search_keyword.lower().replace("_", "-")
            # Return full help if searching for 'all'
            if search_keyword == "all":
                self.epilog = self._json_tip
                return super().format_help()

            # Return group help if searching for a group title
            for group in self._action_groups:
                if group.title and group.title.lower() == search_keyword:
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    formatter.add_text(self._json_tip)
                    return formatter.format_help()

            # Return matched args if searching for an arg name
            matched_actions = []
            for group in self._action_groups:
                for action in group._group_actions:
                    # search option name
                    if any(
                        search_keyword in opt.lower() for opt in action.option_strings
                    ):
                        matched_actions.append(action)
            if matched_actions:
                formatter.start_section(f"Arguments matching '{search_keyword}'")
                formatter.add_arguments(matched_actions)
                formatter.end_section()
                formatter.add_text(self._json_tip)
                return formatter.format_help()

            # No match found
            formatter.add_text(
                f"No group or arguments matching '{search_keyword}'.\n"
                "Use '--help' to see available groups or "
                "'--help=all' to see all available parameters."
            )
            return formatter.format_help()

        # usage
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        formatter.start_section("Config Groups")
        config_groups = ""
        for group in self._action_groups:
            if not group._group_actions:
                continue
            title = group.title
            description = group.description or ""
            config_groups += f"{title: <24}{description}\n"
        formatter.add_text(config_groups)
        formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: Namespace | None = None,
    ):
        if args is None:
            args = sys.argv[1:]

        # Check for --model in command line arguments first
        if args and args[0] == "serve":
            try:
                model_idx = next(
                    i
                    for i, arg in enumerate(args)
                    if arg == "--model" or arg.startswith("--model=")
                )
                logger.warning(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option. "
                    "The `--model` option will be removed in v0.13."
                )

                if args[model_idx] == "--model":
                    model_tag = args[model_idx + 1]
                    rest_start_idx = model_idx + 2
                else:
                    model_tag = args[model_idx].removeprefix("--model=")
                    rest_start_idx = model_idx + 1

                # Move <model> to the front, e,g:
                # [Before]
                # vllm serve -tp 2 --model <model> --enforce-eager --port 8001
                # [After]
                # vllm serve <model> -tp 2 --enforce-eager --port 8001
                args = [
                    "serve",
                    model_tag,
                    *args[1:model_idx],
                    *args[rest_start_idx:],
                ]
            except StopIteration:
                pass

        if "--config" in args:
            args = self._pull_args_from_config(args)

        def repl(match: re.Match) -> str:
            """Replaces underscores with dashes in the matched string."""
            return match.group(0).replace("_", "-")

        # Everything between the first -- and the first .
        pattern = re.compile(r"(?<=--)[^\.]*")

        # Convert underscores to dashes and vice versa in argument names
        processed_args = list[str]()
        for i, arg in enumerate(args):
            if arg.startswith("--help="):
                FlexibleArgumentParser._search_keyword = arg.split("=", 1)[-1].lower()
                processed_args.append("--help")
            elif arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f"{key}={value}")
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith("-O") and arg != "-O" and arg[2] != ".":
                # allow -O flag to be used without space, e.g. -O3 or -Odecode
                # -O.<...> handled later
                # also handle -O=<optimization_level> here
                optimization_level = arg[3:] if arg[2] == "=" else arg[2:]
                processed_args += ["--optimization-level", optimization_level]
            elif (
                arg == "-O"
                and i + 1 < len(args)
                and args[i + 1] in {"0", "1", "2", "3"}
            ):
                # Convert -O <n> to --optimization-level <n>
                processed_args.append("--optimization-level")
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str) -> dict[str, Any]:
            """Creates a nested dictionary from a list of keys and a value.

            For example, `keys = ["a", "b", "c"]` and `value = 1` will create:
            `{"a": {"b": {"c": 1}}}`
            """
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(
            original: dict[str, Any],
            update: dict[str, Any],
        ) -> set[str]:
            """Recursively updates a dictionary with another dictionary.
            Returns a set of duplicate keys that were overwritten.
            """
            duplicates = set[str]()
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    nested_duplicates = recursive_dict_update(original[k], v)
                    duplicates |= {f"{k}.{d}" for d in nested_duplicates}
                elif isinstance(v, list) and isinstance(original.get(k), list):
                    original[k] += v
                else:
                    if k in original:
                        duplicates.add(k)
                    original[k] = v
            return duplicates

        delete = set[int]()
        dict_args = defaultdict[str, dict[str, Any]](dict)
        duplicates = set[str]()
        # Track regular arguments (non-dict args) for duplicate detection
        regular_args_seen = set[str]()
        for i, processed_arg in enumerate(processed_args):
            if i in delete:  # skip if value from previous arg
                continue

            if processed_arg.startswith("--") and "." not in processed_arg:
                if "=" in processed_arg:
                    arg_name = processed_arg.split("=", 1)[0]
                else:
                    arg_name = processed_arg

                if arg_name in regular_args_seen:
                    duplicates.add(arg_name)
                else:
                    regular_args_seen.add(arg_name)
                continue

            if processed_arg.startswith("-") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value_str = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        # False positive, '.' was only in the value
                        continue
                else:
                    value_str = processed_args[i + 1]
                    delete.add(i + 1)

                if processed_arg.endswith("+"):
                    processed_arg = processed_arg[:-1]
                    value_str = json.dumps(list(value_str.split(",")))

                key, *keys = processed_arg.split(".")
                try:
                    value = json.loads(value_str)
                except json.decoder.JSONDecodeError:
                    value = value_str

                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                arg_duplicates = recursive_dict_update(dict_args[key], arg_dict)
                duplicates |= {f"{key}.{d}" for d in arg_duplicates}
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [a for i, a in enumerate(processed_args) if i not in delete]
        if duplicates:
            logger.warning("Found duplicate keys %s", ", ".join(duplicates))

        # Add the dict args back as if they were originally passed as JSON
        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            msg = "Port must be an integer"
            raise ArgumentTypeError(msg) from None

        if not (1024 <= value <= 65535):
            raise ArgumentTypeError("Port must be between 1024 and 65535")

        return value

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tensor-parallel-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count("--config") <= 1, "More than one config file specified!"

        index = args.index("--config")
        if index == len(args) - 1:
            raise ValueError(
                "No config file specified! \
                             Please check your command-line arguments."
            )

        file_path = args[index + 1]

        config_args = self.load_config_file(file_path)

        # 0th index might be the sub command {serve,chat,complete,...}
        # optionally followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0].startswith("-"):
            # No sub command (e.g., api_server entry point)
            args = config_args + args[0:index] + args[index + 2 :]
        elif args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith("-")
            model_in_config = any(arg == "--model" for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file."
                )

            if model_in_cli:
                # Model specified as positional arg, keep CLI version
                args = (
                    [args[0]]
                    + [args[1]]
                    + config_args
                    + args[2:index]
                    + args[index + 2 :]
                )
            else:
                # No model in CLI, use config if available
                args = [args[0]] + config_args + args[1:index] + args[index + 2 :]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2 :]

        return args

    def load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tensor-parallel-size': '4'
            ]
        """
        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml"):
            raise ValueError(
                f"Config file must be of a yaml/yml type. {extension} supplied"
            )

        # only expecting a flat dictionary of atomic types
        processed_args: list[str] = []

        config: dict[str, int | str] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. Check path correctness",
                file_path,
            )
            raise ex

        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    processed_args.append("--" + key)
            elif isinstance(value, list):
                if value:
                    processed_args.append("--" + key)
                    for item in value:
                        processed_args.append(str(item))
            else:
                processed_args.append("--" + key)
                processed_args.append(str(value))

        return processed_args
