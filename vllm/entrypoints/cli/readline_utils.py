# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Readline enhancements for chat interface."""

import contextlib
import readline
from pathlib import Path
from typing import Optional


class ReadlineEnhancer:
    """Enhance readline with history and completion for chat commands."""

    def __init__(self, command_registry=None):
        self.command_registry = command_registry
        self.history_file = Path.home() / ".vllm_chat_history"
        self.commands = []
        self.matches = []  # Initialize matches list

        if readline:
            self._setup_readline()

    def _setup_readline(self):
        """Configure readline for better user experience."""
        # Set up history
        readline.set_history_length(1000)

        # Load history from file
        if self.history_file.exists():
            with contextlib.suppress(Exception):
                readline.read_history_file(str(self.history_file))

        # Set up tab completion
        # Check if we're using GNU readline or libedit (macOS)
        is_libedit = ('libedit' in readline.__doc__
                      if readline.__doc__ else False)

        # Configure tab completion based on readline implementation
        if is_libedit:
            # macOS libedit syntax
            readline.parse_and_bind("bind ^I rl_complete")
            readline.parse_and_bind("bind \\t rl_complete")
        else:
            # GNU readline syntax
            readline.parse_and_bind('tab: complete')
            readline.parse_and_bind('"\t": complete')

        # Also try the standard Python completion binding
        try:
            import rlcompleter
            readline.set_completer(rlcompleter.Completer().complete)
            readline.parse_and_bind("tab: complete")
        except ImportError:
            pass

        # Set completer delimiters to handle paths and commands properly
        readline.set_completer_delims(' \t\n`~!@#$%^&*()-=+[{]}\\|;:\'",<>?')

        # Set the completer function
        readline.set_completer(self._completer)

        # Better key bindings
        readline.parse_and_bind(r'"\e[A": history-search-backward')  # Up arrow
        readline.parse_and_bind(
            r'"\e[B": history-search-forward')  # Down arrow

        # Update available commands
        if self.command_registry:
            self.commands = [
                f"/{cmd}" for cmd in self.command_registry.commands
            ]

    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion function for readline."""
        # This method is called sequentially for each potential match.
        # 'state' is 0 for the first call, 1 for the second, and so on.

        if state == 0:
            # State 0: Generate all possible matches
            self.matches = []

            line_buffer = readline.get_line_buffer()
            parts = line_buffer.split()

            # If we are not typing anything meaningful, do nothing.
            if not parts:
                return None

            # Determine completion context
            # 1. Completing the command itself (e.g., /mo<TAB>)
            if len(parts) == 1 and line_buffer.startswith('/'):
                self.matches = [
                    cmd for cmd in self.commands if cmd.startswith(parts[0])
                ]

            # 2. Completing arguments for a command
            # (e.g., /model <TAB> or/load <TAB>)
            elif len(parts) >= 2 and parts[0] in self.commands:
                cmd_name = parts[0][1:]
                arg_text = parts[1]

                if cmd_name == 'model' and hasattr(self, 'available_models'):
                    self.matches = [
                        m for m in self.available_models
                        if m.startswith(arg_text)
                    ]

                elif cmd_name in ['save', 'load']:
                    try:
                        import glob

                        # Use text which is the current word being completed,
                        # not parts[1]
                        self.matches = glob.glob(text + '*')
                    except Exception:
                        pass

        # Return the match for the current state
        try:
            return self.matches[state]
        except IndexError:
            # No more matches
            return None

    def save_history(self):
        """Save readline history to file."""
        if readline and self.history_file:
            try:
                self.history_file.parent.mkdir(exist_ok=True)
                readline.write_history_file(str(self.history_file))
            except Exception:
                pass

    def set_available_models(self, models: list[str]):
        """Set available models for completion."""
        self.available_models = models

    def add_history(self, line: str):
        """Add a line to history if readline is not available."""
        if not readline and line.strip():
            # Fallback for systems without readline
            try:
                with open(self.history_file, 'a') as f:
                    f.write(line + '\n')
            except Exception:
                pass
