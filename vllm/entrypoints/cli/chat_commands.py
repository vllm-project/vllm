# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enhanced chat command system for vLLM CLI."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


@dataclass
class ChatContext:
    """Context object passed to chat commands."""
    client: OpenAI
    model_name: str
    conversation: list[ChatCompletionMessageParam]
    system_prompt: Optional[str] = None


class ChatCommand(ABC):
    """Base class for chat commands."""

    def __init__(self, name: str, description: str, usage: str = ""):
        self.name = name
        self.description = description
        self.usage = usage or f"/{name}"

    @abstractmethod
    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        """Execute the command and return optional message to display."""
        pass


class CommandRegistry:
    """Registry for chat commands."""

    def __init__(self):
        self.commands: dict[str, ChatCommand] = {}
        self._register_default_commands()

    def register(self, command: ChatCommand) -> None:
        """Register a new command."""
        self.commands[command.name] = command

    def get(self, name: str) -> Optional[ChatCommand]:
        """Get command by name."""
        return self.commands.get(name)

    def is_command(self, text: str) -> bool:
        """Check if text is a command."""
        return text.startswith("/") and len(text) > 1

    def parse_command(self, text: str) -> tuple[Optional[str], list[str]]:
        """Parse command and arguments from text."""
        if not self.is_command(text):
            return None, []

        parts = text[1:].split()
        if not parts:
            return None, []

        cmd_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        return cmd_name, args

    def _register_default_commands(self):
        """Register all default commands."""
        self.register(HelpCommand(self))
        self.register(ClearCommand())
        self.register(SaveCommand())
        self.register(LoadCommand())
        self.register(SystemCommand())
        self.register(ModelCommand())
        self.register(HistoryCommand())
        self.register(RetryCommand())
        self.register(UndoCommand())
        self.register(SettingsCommand())
        # Register both quit and exit to the same command
        quit_cmd = QuitCommand()
        self.register(quit_cmd)
        # Alias for quit
        self.commands["exit"] = quit_cmd


# Command Implementations


class HelpCommand(ChatCommand):
    """Display available commands."""

    def __init__(self, registry: CommandRegistry):
        super().__init__("help", "Show available commands")
        self.registry = registry

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        lines = ["Available commands:"]
        # Handle the edge case where no commands are registered.
        if not self.registry.commands:
            lines.append("  (No commands registered)")
        else:
            # Track command instances
            # to handle aliases (e.g., /quit, /exit) correctly
            # We use id(cmd) to ensure
            # we only process each unique command object once.
            seen_commands = set()
            for name, cmd in sorted(self.registry.commands.items()):
                # Skip if we've already shown this command object
                if id(cmd) in seen_commands:
                    continue
                seen_commands.add(id(cmd))
                # For quit/exit, show both aliases in the usage
                if name in ('quit', 'exit'):
                    usage = "/quit, /exit"
                else:
                    usage = cmd.usage
                lines.append(f"  {usage:<20} - {cmd.description}")
        return "\n".join(lines)


class ClearCommand(ChatCommand):
    """Clear conversation history."""

    def __init__(self):
        super().__init__("clear", "Clear conversation history")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        # Keep system prompt if exists
        context.conversation.clear()
        if context.system_prompt:
            context.conversation.append({
                "role": "system",
                "content": context.system_prompt
            })
        return "Conversation history cleared."


class SaveCommand(ChatCommand):
    """Save conversation to file."""

    def __init__(self):
        super().__init__("save", "Save conversation to file",
                         "/save [filename]")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        filename = args[
            0] if args else f"chat_{datetime.now():%Y%m%d_%H%M%S}.json"

        # Convert conversation to JSON-serializable format
        conversation_data = []
        for msg in context.conversation:
            if isinstance(msg, dict):
                conversation_data.append(msg)
            else:
                # Handle ChatCompletionMessage objects
                conversation_data.append({
                    "role": msg.role,
                    "content": msg.content
                })

        data = {
            "model": context.model_name,
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation_data
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return f"Conversation saved to {filename}"
        except OSError as e:
            return f"Error: Could not save file to '{filename}'. Reason: {e}"


class LoadCommand(ChatCommand):
    """Load conversation from file."""

    def __init__(self):
        super().__init__("load", "Load conversation from file",
                         "/load <filename>")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        if not args:
            return "Usage: /load <filename>"

        filename = args[0]
        if not os.path.exists(filename):
            return f"Error: File not found at '{filename}'"

        try:
            with open(filename, encoding='utf-8') as f:
                data = json.load(f)

        except json.JSONDecodeError:
            return f"Error: '{filename}' is not a valid JSON file."
        except OSError as e:
            return f"Error: Could not read file '{filename}'. Reason: {e}"

        if not isinstance(data, dict):
            return "Error: Invalid format. JSON root must be an object."

        loaded_conversation = data.get("conversation")
        if not isinstance(loaded_conversation, list):
            return "Error: Invalid format. 'conversation' field must be a list."

        loaded_model = data.get("model")
        if not isinstance(loaded_model, str):
            return "Error: Invalid format. 'model' field must be a string."

        # State Update
        # Clear old state
        context.conversation.clear()
        context.system_prompt = None

        # Load new state
        context.conversation.extend(loaded_conversation)
        context.model_name = loaded_model

        # Explicitly check for and set the system prompt from the loaded data
        if context.conversation and context.conversation[0].get(
                "role") == "system":
            context.system_prompt = context.conversation[0].get("content")

        return (f"Successfully loaded conversation from '{filename}'. "
                f"Switched model to '{loaded_model}'.")


class SystemCommand(ChatCommand):
    """Set or view system prompt."""

    def __init__(self):
        super().__init__("system", "Set or view system prompt",
                         "/system [prompt]")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        if not args:
            # Show current system prompt
            if context.system_prompt:
                return f"Current system prompt: {context.system_prompt}"
            return "No system prompt set."

        # Set new system prompt
        new_prompt = " ".join(args)
        context.system_prompt = new_prompt

        # Update in conversation
        if context.conversation:
            first_msg = context.conversation[0]
            # Check if first message is a system message
            if isinstance(first_msg, dict) and first_msg["role"] == "system":
                context.conversation[0]["content"] = new_prompt
            elif hasattr(first_msg, 'role') and first_msg.role == "system":
                # Can't modify ChatCompletionMessage, need to replace it
                context.conversation[0] = {
                    "role": "system",
                    "content": new_prompt
                }
            else:
                # No system message exists, insert one
                context.conversation.insert(0, {
                    "role": "system",
                    "content": new_prompt
                })
        else:
            context.conversation.insert(0, {
                "role": "system",
                "content": new_prompt
            })

        return "System prompt updated."


class ModelCommand(ChatCommand):
    """Switch to a different model or list available models."""

    def __init__(self):
        super().__init__("model", "Switch model", "/model [name]")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        if not args:
            # List available models
            try:
                models = context.client.models.list()
                lines = ["Available models:"]
                for model in models.data:
                    marker = (" (current)"
                              if model.id == context.model_name else "")
                    lines.append(f"  - {model.id}{marker}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error listing models: {e}"

        # Switch model
        new_model = args[0]
        context.model_name = new_model
        return f"Switched to model: {new_model}"


class HistoryCommand(ChatCommand):
    """Display conversation history."""

    def __init__(self):
        super().__init__("history", "Show conversation history",
                         "/history [full]")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        if not context.conversation:
            return "No conversation history."

        # Check if user wants full history without truncation
        show_full = args and args[0] == "full"

        lines = ["Conversation history:"]
        msg_count = 0
        for msg in context.conversation:
            # Handle both dict and ChatCompletionMessage objects
            if isinstance(msg, dict):
                role = msg["role"].upper()
                content = msg["content"]
            else:
                # Handle ChatCompletionMessage objects
                role = msg.role.upper()
                content = msg.content

            # Skip empty messages
            if not content or not content.strip():
                continue

            msg_count += 1
            # Truncate long messages unless showing full
            if not show_full and len(content) > 200:
                content = content[:197] + "..."
            lines.append(f"{msg_count}. [{role}] {content}")

        if msg_count == 0:
            return "No conversation history."

        return "\n".join(lines)


class RetryCommand(ChatCommand):
    """Retry the last assistant response."""

    def __init__(self):
        super().__init__("retry", "Regenerate last response")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:

        # Find the last user message index
        last_user_index = -1
        for i in range(len(context.conversation) - 1, -1, -1):
            msg = context.conversation[i]
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            if role == "user":
                last_user_index = i
                break

        # Check if a retry is possible
        if last_user_index == -1 or last_user_index == len(
                context.conversation) - 1:
            return "Error: No assistant response found to retry."

        # Remove all assistant messages after the last user message
        del context.conversation[last_user_index + 1:]

        # Return a special signal string (can be the last user message content)
        # to notify the main loop that this is a retry action.
        last_user_content = context.conversation[last_user_index].get(
            "content")
        return f"__RETRY__{last_user_content}"


class UndoCommand(ChatCommand):
    """Undo the last conversation turn."""

    def __init__(self):
        super().__init__("undo", "Undo last conversation turn")

    def _get_role(self, msg) -> Optional[str]:
        """Helper function to safely get the role from a message."""
        if isinstance(msg, dict):
            return msg.get("role")
        elif hasattr(msg, 'role'):
            return msg.role
        return None

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        if not context.conversation:
            return "Nothing to undo."

        removed_count = 0

        # Remove the last message if it's from the assistant
        if self._get_role(context.conversation[-1]) == "assistant":
            context.conversation.pop()
            removed_count += 1

        # After potentially removing an assistant message,
        # remove the last message again if it's from the user.
        # This correctly handles a full user-assistant turn.
        if context.conversation and self._get_role(
                context.conversation[-1]) == "user":
            context.conversation.pop()
            removed_count += 1

        if removed_count > 0:
            # Provide more specific feedback
            if removed_count == 1:
                return "Removed the last message."
            else:  # removed_count == 2
                return "Removed the last conversation turn (user + assistant)."

        return "Nothing to undo."


class SettingsCommand(ChatCommand):
    """Display current settings."""

    def __init__(self):
        super().__init__("settings", "Show current settings")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        lines = ["Current settings:"]
        lines.append(f"  Model: {context.model_name}")
        lines.append(f"  API URL: {context.client.base_url}")
        lines.append(f"  Messages in history: {len(context.conversation)}")
        if context.system_prompt:
            lines.append(f"  System prompt: {context.system_prompt[:50]}...")
        return "\n".join(lines)


class QuitCommand(ChatCommand):
    """Quit the chat."""

    def __init__(self):
        super().__init__("quit", "Exit the chat (also /exit)")

    def execute(self, context: ChatContext, args: list[str]) -> Optional[str]:
        # Signal to exit by returning special string
        return "__EXIT__"
