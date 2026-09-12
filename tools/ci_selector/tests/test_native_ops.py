# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The op and wrapper maps, and the csrc droppability pass.

Attribution comes first: giving a file an op it does not really implement is
what authorizes a wrong drop, so the synthetic tests pin who owns what. Then
floors against the live tree, then the pass and its switch through select().
"""

import textwrap

import pytest
import regex as re
from ci_selector.codemap.classify import select
from ci_selector.codemap.native_ops import ENV_VAR, NativeOps, mode
from ci_selector.codemap.step_refs import _source_dep_steps
from helpers import drift_message

NVFP4 = "csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu"
FP4_OPS = {
    "scaled_fp4_quant",
    "scaled_fp4_experts_quant",
    "silu_and_mul_nvfp4_quant",
    "silu_and_mul_scaled_fp4_experts_quant",
}


def _repo(tmp_path, files: dict[str, str]):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return tmp_path


# --- attribution, on synthetic trees ---------------------------------------


def test_a_registration_tu_owns_its_ops(tmp_path):
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/bindings.cpp": """
            void impl_fn(int x) { return; }
            ops.def("my_op(Tensor a) -> ()");
            ops.impl("my_op", &impl_fn);
        """,
            },
        )
    )
    assert no.file_ops["csrc/bindings.cpp"] == frozenset({"my_op"})


def test_a_caller_does_not_inherit_its_callees_op(tmp_path):
    """A file that only calls an impl symbol must not receive its op."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/bindings.cpp": """
            void helper_fn(int x) { return; }
            ops.def("helper_op(Tensor a) -> ()");
            ops.impl("helper_op", &helper_fn);
        """,
                "csrc/kernel.cu": """
            void kernel_main(int y) {
              helper_fn(y);
            }
        """,
            },
        )
    )
    assert "csrc/kernel.cu" not in no.file_ops


def test_a_callee_inherits_its_callers_ops(tmp_path):
    """The entry file defines the impl symbol and calls a symbol only the
    kernel file defines, so the kernel file reaches the op."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/entry.cu": """
            void fp_entry(int x) {
              fp_kernel_launch(x);
            }
            ops.def("fp_op(Tensor a) -> ()");
            ops.impl("fp_op", &fp_entry);
        """,
                "csrc/kernel.cu": """
            void fp_kernel_launch(int y) { return; }
        """,
            },
        )
    )
    assert no.file_ops.get("csrc/kernel.cu") == frozenset({"fp_op"})


def test_a_shared_helper_name_does_not_smear_ops(tmp_path):
    """A name defined in two files routes nothing."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/a.cu": """
            void round_helper(int x) { return; }
            ops.def("op_a(Tensor a) -> ()");
            ops.impl("op_a", &round_helper_a);
            void round_helper_a(int x) { round_helper(x); }
        """,
                "csrc/b.cu": """
            void round_helper(int x) { return; }
        """,
            },
        )
    )
    assert "csrc/b.cu" not in no.file_ops


def test_a_header_inherits_its_including_tus_ops(tmp_path):
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/entry.cu": """
            #include "shapes.h"
            void op_entry(int x) { return; }
            ops.def("hdr_op(Tensor a) -> ()");
            ops.impl("hdr_op", &op_entry);
        """,
                "csrc/shapes.h": "\n",
            },
        )
    )
    assert no.file_ops.get("csrc/shapes.h") == frozenset({"hdr_op"})


def test_namespaces_derive_from_cmake_and_join_wrappers(tmp_path):
    """The full loop: a cmake target names the namespace (on the line after
    the opener, as every real call site spells it), the csrc file registers
    the op, and the vllm wrapper joins through the derived namespace."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "CMakeLists.txt": """
            define_extension_target(
              _demo_C
              DESTINATION vllm
              LANGUAGE CXX)
        """,
                "csrc/bindings.cpp": """
            void impl_fn(int x) { return; }
            ops.def("demo_op(Tensor a) -> ()");
            ops.impl("demo_op", &impl_fn);
        """,
                "vllm/_ops.py": """
            import torch

            def demo_op(a):
                return torch.ops._demo_C.demo_op(a)
        """,
            },
        )
    )
    assert "_demo_C" in no.namespaces
    assert no.wrappers.get("demo_op") == frozenset({("vllm/_ops.py", "demo_op")})


def test_commented_build_files_mint_no_namespace(tmp_path):
    """A commented-out target or registration must not widen the set: a
    foreign name that vllm/ happens to reference would join a wrong
    wrapper."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "CMakeLists.txt": """
            # define_extension_target(_ghost_C DESTINATION vllm)
            define_extension_target(
              _real_C
              DESTINATION vllm)
        """,
                "csrc/bindings.cpp": """
            // TORCH_LIBRARY(_ghost2_C, m) {}
            STABLE_TORCH_LIBRARY(_lit_C, m) { }
            void f(int x) { return; }
            ops.def("an_op(Tensor a) -> ()");
            ops.impl("an_op", &f);
        """,
            },
        )
    )
    assert "_ghost_C" not in no.namespaces
    assert "_ghost2_C" not in no.namespaces
    assert {"_real_C", "_lit_C"} <= no.namespaces


def test_macro_placeholders_are_not_namespaces(tmp_path):
    """registration.h's own macro definitions pass NAME/TORCH_EXTENSION_NAME/
    CONCAT through the registration regexes; none is a namespace."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/registration.h": """
            #define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
            #define REGISTER_EXTENSION(NAME) PyMODINIT_FUNC CONCAT(PyInit_, NAME)()
        """,
                "csrc/bindings.cpp": """
            TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _sub), ops) { }
            void f(int x) { return; }
            ops.def("an_op(Tensor a) -> ()");
            ops.impl("an_op", &f);
        """,
                "CMakeLists.txt": """
            define_extension_target(
              _base_C
              DESTINATION vllm)
        """,
            },
        )
    )
    assert not no.namespaces & {"NAME", "TORCH_EXTENSION_NAME", "CONCAT"}
    assert "_base_C_sub" in no.namespaces  # CONCAT suffix expands over targets


def test_no_csrc_tree_fails_open(tmp_path):
    no = NativeOps.build(tmp_path)
    assert no.file_ops == {}
    assert no.error


def test_mode_defaults_on_and_rejects_typos(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert mode() == "on"
    monkeypatch.setenv(ENV_VAR, "off")
    assert mode() == "off"
    monkeypatch.setenv(ENV_VAR, "maybe")
    with pytest.raises(ValueError):
        mode()


def test_ownership_excludes_the_exclusive_trees():
    no = NativeOps()
    assert no.owns("csrc/attention/foo.cu")
    assert not no.owns("csrc/cpu/foo.cpp")
    assert not no.owns("csrc/rocm/foo.cu")
    assert not no.owns("cmake/foo.cmake")
    assert not no.owns("vllm/foo.py")


# --- live drift floors ------------------------------------------------------


@pytest.fixture(scope="session")
def live_ops(vllm_repo):
    return NativeOps.build(vllm_repo, [])


@pytest.mark.drift
def test_the_registration_parse_still_finds_the_ops(live_ops):
    assert live_ops.error is None, live_ops.error
    assert live_ops.op_count >= 200, drift_message(
        f"the registration parse found {live_ops.op_count} ops, below the "
        "200 floor (measured basis: 222, zero unparsed)",
        "Every op the parse misses costs its files their droppability, so "
        "the channel quietly stops doing anything.",
        "a new registration macro style landed in csrc: teach the regexes in "
        "ci_selector/codemap/native_ops.py the new shape",
    )
    wrapped = sum(1 for op in live_ops.wrappers if live_ops.wrappers[op])
    assert wrapped >= 150, drift_message(
        f"only {wrapped} ops reach a Python wrapper (floor 150; basis 220)",
        "An op with no wrapper blocks its whole file.",
        "the wrapper call sites moved, or the namespace derivation "
        "(_derive_namespaces in ci_selector/codemap/native_ops.py) stopped "
        "seeing the registering namespaces",
    )


@pytest.mark.drift
def test_the_namespace_derivation_holds_its_anchors(live_ops):
    """The four namespaces nothing should ever dislodge, plus a collapse
    floor: losing the cmake leg alone leaves ~9 csrc literals."""
    anchor = {"_C", "_moe_C", "_rocm_C", "cumem_allocator"}
    assert anchor <= live_ops.namespaces, drift_message(
        f"derived namespaces lost {sorted(anchor - live_ops.namespaces)}",
        "Wrappers in a lost namespace stop joining, so their files silently "
        "become permanently non-droppable.",
        "define_extension_target or the registration macros moved: update "
        "_derive_namespaces in ci_selector/codemap/native_ops.py",
    )
    assert len(live_ops.namespaces) >= 12, drift_message(
        f"only {len(live_ops.namespaces)} namespaces derived (floor 12, basis 29)",
        "A collapsed set means a whole build-file leg stopped parsing.",
        "check the cmake/csrc legs of _derive_namespaces in native_ops.py",
    )


# Namespaces vllm/ references that nothing in-tree defines. Three kinds:
# upstream torch, third-party packages, and vLLM-ecosystem extensions built
# outside this repo (vllm-xpu-kernels; the vendored vllm-flash-attn
# subproject's install components). The derivation can never produce these,
# and the detector below fires when a NEW namespace appears in neither set.
_EXTERNAL_NS = {
    "aten",
    "symm_mem",
    "higher_order",
    "zentorch",
    "aiter",
    "oink",
    "fbgemm",
    "_xpu_C",
    "_vllm_fa2_C",
    "_vllm_fa3_C",
}
_PY_LIB = re.compile(r'\bLibrary\(\s*"(\w+)"')
_NS_REF = re.compile(r"\btorch\s*\.\s*ops\s*\.\s*(\w+)\s*\.")


@pytest.mark.drift
def test_every_torch_ops_namespace_is_accounted_for(vllm_repo, live_ops):
    """The per-namespace detector the wrapped>=150 mass floor cannot be: one
    new unaccounted namespace means its kernels never join and their files
    are silently never droppable, which the mass floor cannot see."""
    referenced: set[str] = set()
    py_libs: set[str] = set()
    for p in (vllm_repo / "vllm").rglob("*.py"):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        referenced.update(_NS_REF.findall(text))
        py_libs.update(_PY_LIB.findall(text))
    assert len(referenced) >= 15, drift_message(
        f"the vllm/ sweep found only {len(referenced)} torch.ops namespaces "
        "(floor 15, basis 19)",
        "An empty enumeration would make this detector read clean forever.",
        "the torch.ops call shape moved: fix _NS_REF in this test",
    )
    unaccounted = referenced - live_ops.namespaces - py_libs - _EXTERNAL_NS
    assert not unaccounted, drift_message(
        f"torch.ops namespaces nothing accounts for: {sorted(unaccounted)}",
        "If it is a new in-tree extension its ops will never join a wrapper "
        "and its files are silently never droppable; if it is external, "
        "selection is fine but this detector is now blind to the next one.",
        "in-tree: check _derive_namespaces sees its build files; external: "
        "add it to _EXTERNAL_NS here with its origin",
    )


@pytest.mark.drift
def test_every_registering_file_yields_ops(vllm_repo, live_ops):
    """A file containing `.def(\"` must derive an op, else a registration
    style stopped parsing."""
    bad = []
    for p in (vllm_repo / "csrc").rglob("*"):
        if p.is_file() and p.suffix in (".cu", ".cpp", ".cc", ".c"):
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            rel = p.relative_to(vllm_repo).as_posix()
            if '.def("' in text and rel not in live_ops.file_ops:
                bad.append(rel)
    assert not bad, drift_message(
        f"registering files derive no ops: {bad[:5]}",
        "Their changes lose all proxy evidence and any droppability the "
        "channel would have granted elsewhere on the same diff is suspect.",
        "a new .def style: extend the regexes in native_ops.py",
    )


@pytest.mark.drift
def test_the_nvfp4_attribution_anchor(live_ops):
    """The kernel file inherits exactly the entry file's fp4 ops: never a
    helper's op, never the whole registry."""
    got = live_ops.file_ops.get(NVFP4, frozenset())
    assert got == frozenset(FP4_OPS), drift_message(
        f"{NVFP4} derives {sorted(got) or 'NOTHING'}, expected the 4 fp4 ops",
        "This is the 50230 safety anchor: wrong attribution here is exactly "
        "the shape that once authorized dropping two actually-failed jobs.",
        "the fp4 entry/kernel split moved: re-derive and update the anchor",
        "the linkage legs in native_ops.py regressed: fix the attribution",
    )
    assert "get_device_attribute" not in got


# --- the droppability pass, through the real select() -----------------------


def test_a_joined_tu_gets_droppable_steps_with_proxy_paths(state):
    sel = select(state, [NVFP4])
    assert len(sel.claims) == 1
    claim = sel.claims[0]
    assert claim.droppable_step_ids, "a fully wrapped file granted nothing"
    assert claim.droppable_step_ids <= claim.step_ids
    assert "vllm/_custom_ops.py" in claim.evidence_paths
    declared = _source_dep_steps(state, NVFP4)
    assert not claim.droppable_step_ids & declared
    producers = {s for ss in state.artifacts.producers_of.values() for s in ss}
    builders = {s for ss in state.artifacts.self_builders.values() for s in ss}
    assert not claim.droppable_step_ids & (producers | builders)
    droppable = next(iter(claim.droppable_step_ids))
    recorded = sel.selected_paths.get(droppable)
    assert recorded and any(r and "vllm/_custom_ops.py" in r for r in recorded), (
        "proxy paths not recorded for a droppable step"
    )


def test_direct_caller_test_steps_stay_non_droppable(state):
    """A test naming one of the file's ops may call it from a frame the
    recorder cannot see."""
    sel = select(state, [NVFP4])
    claim = next(c for c in sel.claims if c.droppable_step_ids)
    held = set()
    from ci_selector.codemap.classify import _steps_targeting

    for tf in state.native_ops.test_files_for(NVFP4):
        held |= _steps_targeting(state, tf)
    assert not claim.droppable_step_ids & held


def test_the_switch_off_disarms_both_routing_and_droppability(state, monkeypatch):
    """Off means the op parse is not trusted: nothing becomes droppable and
    the routing rule declines. The fallback is the build-map-scoped fail-open
    — family evidence from the cmake guards, independent of the parse the
    switch just disarmed — so the off answer is no longer a superset: it
    keeps the file's device families and sheds the cross-family steps the
    op-name join over-reaches into (contract re-cut 2026-09-02, see knobs)."""
    on = select(state, [NVFP4])
    monkeypatch.setenv(ENV_VAR, "off")
    off = select(state, [NVFP4])
    assert off.selected and not off.run_all
    assert any(
        c.rule == "fail-open" and "build-map scoped" in c.detail for c in off.claims
    )
    assert not any(c.droppable_step_ids for c in off.claims)
    assert any(c.droppable_step_ids for c in on.claims)


def test_an_unmapped_csrc_path_grants_nothing(state):
    sel = select(state, ["csrc/attention/attention_kernels.cu"])  # ghost
    assert not any(c.evidence_paths for c in sel.claims)


def test_droppable_stays_a_subset_after_every_pass(state):
    """The check at construction cannot see later mutation."""
    for paths in ([NVFP4], ["csrc/libtorch_stable/cache_kernels.cu"]):
        sel = select(state, paths)
        for c in sel.claims:
            assert c.droppable_step_ids <= c.step_ids, c.rule


# --- the decide-side proxy append ------------------------------------------


def test_append_op_proxies_gates_unrecorded_names(monkeypatch, tmp_path):
    from ci_selector import decide as decide_mod
    from ci_selector.coverage.changed_funcs import Attribution, FileQuery, Query

    no = NativeOps(
        file_ops={"csrc/k.cu": frozenset({"my_op"})},
        wrappers={"my_op": frozenset({("vllm/w.py", "my_op")})},
    )

    class FakeState:
        native_ops = no

    monkeypatch.setattr(
        "ci_selector.codemap.worktree.state_for", lambda repo, base: FakeState()
    )

    class FakeTable:
        unfaithful_paths: set = set()

    def run(union):
        q = Query(
            base="b",
            head="h",
            files=[
                FileQuery(
                    path="csrc/k.cu",
                    status=Attribution.NAMELESS,
                    in_recorder_scope=False,
                ),
            ],
        )
        decide_mod._append_op_proxies(q, tmp_path, "b", union, FakeTable())
        return q

    q = run({"vllm/w.py": frozenset({"my_op"})})
    proxy = [f for f in q.files if f.proxy]
    assert len(proxy) == 1
    assert proxy[0].status is Attribution.ATTRIBUTED
    assert proxy[0].head_names == frozenset({"my_op"})

    q = run({"vllm/w.py": frozenset({"other"})})
    proxy = [f for f in q.files if f.proxy]
    assert proxy[0].status is Attribution.FAILED  # unrecorded derived name
    assert proxy[0].fail_open

    # a non-csrc diff appends nothing and never builds state
    q2 = Query(
        base="b",
        head="h",
        files=[
            FileQuery(path="vllm/mod.py", status=Attribution.ATTRIBUTED),
        ],
    )
    decide_mod._append_op_proxies(q2, tmp_path, "b", {}, FakeTable())
    assert not any(f.proxy for f in q2.files)


def test_commented_registrations_mint_no_phantom_ops(tmp_path):
    """An op that exists only inside a comment reaches no wrapper, and one
    such op blocks its file and every header including it."""
    no = NativeOps.build(
        _repo(
            tmp_path,
            {
                "csrc/bindings.cpp": """
            // ops.def("phantom_op(Tensor a) -> ()");
            /* ops.impl("phantom_two", &gone_fn); */
            void real_fn(int x) { return; }
            ops.def("real_op(Tensor a) -> ()");
            ops.impl("real_op", &real_fn);
        """,
            },
        )
    )
    assert no.file_ops["csrc/bindings.cpp"] == frozenset({"real_op"})
    assert no.op_count == 1
