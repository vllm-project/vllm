# SPDX-License-Identifier: Apache-2.0
"""MP coordinator for LMCache multi-process (mp) servers.

A standalone FastAPI process that mp servers across nodes register with, so they
can be coordinated as a fleet: state reconcile (e.g. quota persistence and
broadcast on join), blend-lookup routing across model replicas, and KV-cache
operations (pin, prefetch, ...).

The coordinator exposes a REST API; mp servers register / heartbeat / deregister
over HTTP, and the coordinator pushes commands back to each mp server's own HTTP
server. The package currently ships the framework -- the FastAPI app with
auto-discovered ``http_apis`` routers, an instance registry, and a health-check
loop -- plus the ``/instances`` membership resource. Further capabilities are
added as new ``http_apis`` routers without framework changes; to push to an mp
server, a router resolves its address from the registry and POSTs to that
server's specific endpoint.
"""
