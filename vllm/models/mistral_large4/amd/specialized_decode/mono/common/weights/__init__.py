# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side weight preparation every mono kernel needs: the storage
vocabulary (``config``), MX format conversion (``formats``), and the packing
that turns a loaded tensor into what a kernel reads (``packing``, ``layer``).
"""
