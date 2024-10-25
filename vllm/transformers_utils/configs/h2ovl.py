# Adapted from
# https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/configuration_h2ovl_chat.py
# --------------------------------------------------------
# H2OVL
# Copyright (c) 2024 H2O.ai
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------

from .internvl import InternVLChatConfig

class H2OVLChatConfig(InternVLChatConfig):
    model_type = "h2ovl_chat"