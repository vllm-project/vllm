# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [CN] 文件总览：OpenAI 兼容服务的历史入口，如今只是一个「废弃的兼容壳」
# [CN] 职责：把 launchers/ 下拆分好的装配函数原样 re-export，本文件自身没有任何启动逻辑，
# [CN]       内容只有 import、__all__ 和两条 DeprecationWarning。
# [CN] 链路：python -m vllm.entrypoints.openai.api_server
# [CN]        -> 本文件 __main__ -> launchers/api_server/entry.py:main()
# [CN]        -> build_async_engine_client()（起引擎）-> build_app()（装路由）
# [CN]        -> setup_server()（起 HTTP/unix socket 服务）
# [CN] 真正的实现分三处：
# [CN]   launchers/app.py                  —— FastAPI app 与中间件装配
# [CN]   launchers/api_server/routers.py   —— /v1/chat/completions 等路由注册
# [CN]   launchers/launcher.py             —— socket 创建与多进程 worker 启动
# [CN] 易错点：模块被 import 的瞬间就发废弃警告（stacklevel=1 指向调用方），
# [CN]         看到它不是你的调用方式有问题，而是这个入口整体已经迁移到 launchers。

import warnings

from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.launchers.api_server.entry import (
    build_and_serve,
    build_async_engine_client,
    build_async_engine_client_from_engine_args,
    run_server,
    run_server_worker,
)
from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.launchers.launcher import (
    create_server_socket,
    create_server_unix_socket,
    setup_server,
    validate_api_server_args,
)
from vllm.entrypoints.launchers.render.app_state import init_render_app_state
from vllm.entrypoints.launchers.render.entry import build_and_serve_renderer

warnings.warn(
    "`vllm.entrypoints.openai.api_server` is deprecated and will likely be"
    "unsupported in a future version. Use the corresponding function from "
    "`vllm.entrypoints.launchers` instead.",
    DeprecationWarning,
    stacklevel=1,
)


# [CN] __all__ 是纯 re-export 清单，用来维持「from api_server import build_app」这类旧写法。
# [CN] 代价是 import 本模块必然触发上面的废弃警告 —— 新代码应直接从 launchers 包导入。

__all__ = [
    "build_async_engine_client",
    "build_async_engine_client_from_engine_args",
    "build_app",
    "init_app_state",
    "init_render_app_state",
    "create_server_socket",
    "create_server_unix_socket",
    "validate_api_server_args",
    "setup_server",
    "build_and_serve",
    "build_and_serve_renderer",
    "run_server",
    "run_server_worker",
    "register_api_routers",
]

# [CN] 旧命令 python -m ...api_server 仍保留，官方入口已改为 `vllm server`。
# [CN] main 的 import 故意延后到分支内部：不这么做的话，任何 import 本模块的第三方
# [CN] 都会被连带拉起整棵 serving 依赖树（含 HTTP 栈），而它们其实只要那几个 re-export。

if __name__ == "__main__":
    warnings.warn(
        "The `python -m vllm.entrypoints.openai.api_server` command is deprecated "
        "and may be removed in a future release. Please use `vllm server` instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    from vllm.entrypoints.launchers.api_server.entry import main

    main()
