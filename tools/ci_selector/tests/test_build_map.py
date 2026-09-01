# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The build map: which device families CMake says compile a file.

Three layers, in file order: the walker's grammar on synthetic trees, anchors
against the live checkout so a moved guard is loud, and the narrowing through
the real `select()`.
"""

import textwrap

import pytest
from ci_selector.codemap import unions
from ci_selector.codemap.build_map import ALL_FAMILIES, ENV_VAR, BuildMap, mode
from ci_selector.codemap.classify import select
from ci_selector.codemap.step_refs import _source_dep_steps
from helpers import drift_message

CUDA_TU = "csrc/libtorch_stable/quantization/machete/machete_pytorch.cu"
SHARED_TU = "csrc/libtorch_stable/cache_kernels.cu"


def _mini_repo(tmp_path, cmake_text, files=(), extra_cmake=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "CMakeLists.txt").write_text(textwrap.dedent(cmake_text))
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    for rel, text in (extra_cmake or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return tmp_path


# --- the grammar, on synthetic trees --------------------------------------


def test_cuda_guard_maps_cuda(tmp_path):
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(VLLM_GPU_LANG STREQUAL "CUDA")
          list(APPEND SRC "csrc/a.cu")
        endif()
    """,
        )
    )
    assert bm.families["csrc/a.cu"] == frozenset({"cuda"})


def test_cuda_or_hip_guard_maps_both(tmp_path):
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
          list(APPEND SRC "csrc/a.cu")
        endif()
    """,
        )
    )
    assert bm.families["csrc/a.cu"] == frozenset({"cuda", "amd"})


def test_unguarded_literal_is_unmapped(tmp_path):
    """Every family compiles it, so there is nothing to scope with."""
    bm = BuildMap.build(_mini_repo(tmp_path, 'list(APPEND SRC "csrc/a.cpp")\n'))
    assert "csrc/a.cpp" not in bm.families


def test_unknown_condition_inherits_the_enclosing_scope(tmp_path):
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(VLLM_GPU_LANG STREQUAL "CUDA")
          if(SOME_UNKNOWN_FLAG)
            list(APPEND SRC "csrc/a.cu")
          endif()
        endif()
        if(SOME_UNKNOWN_FLAG)
          list(APPEND SRC "csrc/b.cu")
        endif()
    """,
        )
    )
    assert bm.families["csrc/a.cu"] == frozenset({"cuda"})
    assert "csrc/b.cu" not in bm.families  # inherits ALL -> unmapped


def test_else_of_a_known_condition_takes_the_complement(tmp_path):
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(VLLM_GPU_LANG STREQUAL "CUDA")
          list(APPEND SRC "csrc/a.cu")
        else()
          list(APPEND SRC "csrc/b.cpp")
        endif()
    """,
        )
    )
    assert bm.families["csrc/b.cpp"] == ALL_FAMILIES - {"cuda"}


def test_else_of_an_unknown_condition_fails_open(tmp_path):
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(SOME_UNKNOWN_FLAG)
          list(APPEND SRC "csrc/a.cu")
        else()
          list(APPEND SRC "csrc/b.cpp")
        endif()
    """,
        )
    )
    assert "csrc/b.cpp" not in bm.families


def test_return_under_a_recognized_guard_narrows_the_rest_of_the_file(tmp_path):
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
           NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
          return()
        endif()
        list(APPEND SRC "csrc/a.cu")
    """,
        )
    )
    assert bm.families["csrc/a.cu"] == frozenset({"cuda", "amd"})


def test_return_under_an_unknown_guard_subtracts_nothing(tmp_path):
    """The one move that could flip fail-open into under-selection."""
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(SOME_UNKNOWN_FLAG)
          return()
        endif()
        list(APPEND SRC "csrc/a.cu")
    """,
        )
    )
    assert "csrc/a.cu" not in bm.families  # still ALL, not ALL-minus-unknown


def test_an_included_cmake_file_inherits_its_include_site(tmp_path):
    repo = _mini_repo(
        tmp_path,
        """
        if(VLLM_TARGET_DEVICE STREQUAL "cpu")
          include(${CMAKE_CURRENT_LIST_DIR}/cmake/sub.cmake)
        endif()
    """,
        extra_cmake={"cmake/sub.cmake": 'list(APPEND SRC "csrc/x.cpp")\n'},
    )
    bm = BuildMap.build(repo)
    assert bm.families["cmake/sub.cmake"] == frozenset({"cpu"})
    assert bm.families["csrc/x.cpp"] == frozenset({"cpu"})


def test_glob_is_evaluated_against_the_tree(tmp_path):
    repo = _mini_repo(
        tmp_path,
        """
        if(VLLM_GPU_LANG STREQUAL "HIP")
          file(GLOB K "csrc/gen/*.cu")
        endif()
    """,
        files=["csrc/gen/k1.cu", "csrc/gen/k2.cu"],
    )
    bm = BuildMap.build(repo)
    assert bm.families["csrc/gen/k1.cu"] == frozenset({"amd"})
    assert bm.families["csrc/gen/k2.cu"] == frozenset({"amd"})


def test_a_missing_cmakelists_fails_open(tmp_path):
    bm = BuildMap.build(tmp_path)
    assert bm.families == {}
    assert bm.error


def test_a_header_inherits_its_including_tus_families(tmp_path):
    repo = _mini_repo(
        tmp_path,
        """
        if(VLLM_GPU_LANG STREQUAL "CUDA")
          list(APPEND SRC "csrc/a.cu")
        endif()
    """,
    )
    (repo / "csrc").mkdir(exist_ok=True)
    (repo / "csrc/a.cu").write_text('#include "x.h"\n')
    (repo / "csrc/x.h").write_text("")
    bm = BuildMap.build(repo)
    assert bm.families["csrc/x.h"] == frozenset({"cuda"})


def test_an_unreachable_header_falls_back_to_directory_affinity(tmp_path):
    repo = _mini_repo(
        tmp_path,
        """
        if(VLLM_TARGET_DEVICE STREQUAL "cpu")
          list(APPEND SRC "csrc/sub/a.cpp")
        endif()
    """,
    )
    (repo / "csrc/sub").mkdir(parents=True, exist_ok=True)
    (repo / "csrc/sub/a.cpp").write_text("")
    (repo / "csrc/sub/orphan.h").write_text("")
    bm = BuildMap.build(repo)
    assert bm.families["csrc/sub/orphan.h"] == frozenset({"cpu"})


def test_headers_stand_down_as_a_group_when_the_closure_goes_dark(tmp_path):
    """Past the unresolved bound every header is dropped, sources kept."""
    repo = _mini_repo(
        tmp_path,
        """
        if(VLLM_GPU_LANG STREQUAL "CUDA")
          list(APPEND SRC "csrc/a.cu")
        endif()
    """,
    )
    (repo / "csrc/lost").mkdir(parents=True, exist_ok=True)
    (repo / "csrc/a.cu").write_text('#include "x.h"\n')
    (repo / "csrc/x.h").write_text("")
    for i in range(6):
        (repo / f"csrc/lost/h{i}.h").write_text("")
    bm = BuildMap.build(repo)
    assert bm.unresolved_headers == 6
    assert "csrc/x.h" not in bm.families
    assert bm.families["csrc/a.cu"] == frozenset({"cuda"})


def test_a_dropped_guard_widens_never_narrows(tmp_path):
    """A guard we cannot read must leave the file unmapped, never narrow."""
    guarded = """
        if(VLLM_GPU_LANG STREQUAL "CUDA")
          list(APPEND SRC "csrc/a.cu")
        endif()
    """
    corrupted = guarded.replace(
        'VLLM_GPU_LANG STREQUAL "CUDA"', "SOMETHING_UNRECOGNIZED"
    )
    assert BuildMap.build(_mini_repo(tmp_path / "g", guarded)).families[
        "csrc/a.cu"
    ] == frozenset({"cuda"})
    assert (
        "csrc/a.cu"
        not in BuildMap.build(_mini_repo(tmp_path / "c", corrupted)).families
    )


# --- the knob --------------------------------------------------------------


def test_mode_defaults_on_and_rejects_typos(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert mode() == "on"
    monkeypatch.setenv(ENV_VAR, "off")
    assert mode() == "off"
    monkeypatch.setenv(ENV_VAR, "offf")
    with pytest.raises(ValueError):
        mode()


# --- drift anchors against the live checkout -------------------------------


@pytest.fixture(scope="session")
def live_map(vllm_repo):
    return BuildMap.build(vllm_repo)


@pytest.mark.drift
def test_the_walker_still_reads_the_real_guards(live_map):
    """One anchor per grammar leg, against the live CMakeLists.txt."""
    anchors = {
        SHARED_TU: frozenset({"cuda", "amd"}),  # shared source list
        CUDA_TU: frozenset({"cuda"}),  # named inside the CUDA-only block
        "cmake/cpu_extension.cmake": frozenset({"cpu"}),  # forward block
        "cmake/external_projects/flashkda.cmake": frozenset({"cuda"}),
    }
    wrong = {
        p: (sorted(live_map.families.get(p, set())) or "UNMAPPED", sorted(want))
        for p, want in anchors.items()
        if live_map.families.get(p) != want
    }
    assert not wrong, drift_message(
        f"the CMake walker no longer derives the anchor families: {wrong}",
        "An unmapped anchor means these files went back to selecting every "
        "step on every image.",
        "a guard or source list moved in CMakeLists.txt: teach the walker "
        "grammar in ci_selector/codemap/build_map.py the new shape",
        "the file itself moved or was deleted: update the anchor here",
    )


@pytest.mark.drift
def test_the_map_has_not_collapsed(live_map):
    """A walker that maps nothing fails open everywhere and reads as clean."""
    got = sum(1 for p in live_map.families if p.startswith("csrc/"))
    assert got >= 100, drift_message(
        f"the build map holds {got} csrc entries, below the 100 floor",
        "Below the floor the narrowing has stopped and every csrc change "
        "selects every step on every image again.",
        "CMakeLists.txt was restructured: extend the walker grammar in "
        "ci_selector/codemap/build_map.py",
    )
    assert live_map.error is None, live_map.error
    assert live_map.total_headers and (
        live_map.unresolved_headers / live_map.total_headers <= 0.2
    ), f"{live_map.unresolved_headers}/{live_map.total_headers} headers dark"


@pytest.mark.drift
def test_the_map_agrees_with_the_exclusive_namespaces(live_map):
    """csrc/cpu and csrc/rocm are pinned in handwritten.py, so the derived
    map must agree. Headers only widen, so check source files."""
    bad = {
        p: sorted(f)
        for p, f in live_map.families.items()
        if p.endswith((".cu", ".cpp", ".cc", ".c", ".hip"))
        and (
            (p.startswith("csrc/cpu/") and f != frozenset({"cpu"}))
            or (p.startswith("csrc/rocm/") and f != frozenset({"amd"}))
        )
    }
    assert not bad, drift_message(
        f"derived families disagree with EXCLUSIVE_NAMESPACES: {bad}",
        "Two mechanisms now claim different scopes for the same tree; one of "
        "them is silently wrong about which suites a kernel change can break.",
        "the build moved sources across device trees: fix whichever side is "
        "stale (handwritten.py EXCLUSIVE_NAMESPACES or the walker grammar)",
    )


@pytest.mark.drift
def test_csrc_data_readers_keep_their_steps(vllm_repo, state, live_map):
    """Two tests read csrc files as data at runtime. Who compiles a file
    says nothing about who reads it, so those steps have to survive."""
    readers = {
        "tests/kernels/test_bf16_skinny_gemm.py": (
            "csrc/libtorch_stable/dsv3_fused_a_gemm.cu"
        ),
        "tests/kernels/quantization/test_rdna3_compile_guards.py": (
            "csrc/rocm/torch_bindings.cpp"
        ),
    }
    live = {t: p for t, p in readers.items() if (vllm_repo / t).is_file()}
    assert len(live) >= 2, drift_message(
        f"only {len(live)} of the pinned csrc data-reader tests still exist",
        "The reader pins are the only guard against family scoping dropping "
        "a step whose test parses csrc content as data.",
        "the test moved: update the reader table here",
    )
    for test_file, csrc_path in live.items():
        sel = select(state, [csrc_path])
        holders = {
            s.step_id
            for p in state.pipelines
            for s in p.steps
            if s.step_id in state.auto_step_ids
            and s.step_id in sel.selected
            and test_file in (getattr(p.targets.get(s.step_id), "haystack", "") or "")
        }
        covered = {
            s.step_id
            for p in state.pipelines
            for s in p.steps
            if s.step_id in state.auto_step_ids
            and test_file in (getattr(p.targets.get(s.step_id), "haystack", "") or "")
        }
        assert not covered or holders, (
            f"{csrc_path} no longer selects any step running {test_file}, "
            f"which parses it as data ({sorted(covered)[:3]} dropped)"
        )


# --- the narrowing, through the real select() ------------------------------


def test_a_cuda_only_tu_sheds_the_other_families_steps(state):
    """Only a step naming csrc/ in its own source deps may survive outside
    the mapped families, since that is CI's own trigger."""
    sel = select(state, [CUDA_TU])
    per, union, nonfamily = state.family_partition()
    picked = set(sel.selected)
    declared = _source_dep_steps(state, CUDA_TU)
    assert picked & per["xpu"] <= declared, sorted(picked & per["xpu"] - declared)[:5]
    assert picked & per["amd"] <= declared, sorted(picked & per["amd"] - declared)[:5]
    assert len(picked & nonfamily) > 100  # the main-image world stays


def test_a_shared_cuda_amd_tu_keeps_the_amd_leg(state):
    sel = select(state, [SHARED_TU])
    per, _union, nonfamily = state.family_partition()
    picked = set(sel.selected)
    declared = _source_dep_steps(state, SHARED_TU)
    assert picked & per["amd"]
    assert picked & per["xpu"] <= declared, sorted(picked & per["xpu"] - declared)[:5]
    assert len(picked & nonfamily) > 100


def test_an_unmapped_csrc_path_keeps_the_full_union(state, monkeypatch):
    """A path the base tree does not hold, deleted or added by the PR, must
    keep the wider answer."""
    ghost = "csrc/attention/attention_kernels.cu"
    assert ghost not in state.build_map.families
    on = select(state, [ghost]).selected
    monkeypatch.setenv(ENV_VAR, "off")
    off = select(state, [ghost]).selected
    assert on == off


def test_the_knob_off_restores_the_blanket(state, monkeypatch):
    on = select(state, [CUDA_TU]).selected
    monkeypatch.setenv(ENV_VAR, "off")
    off = select(state, [CUDA_TU]).selected
    assert set(on) < set(off)
    per, _union, _nonfamily = state.family_partition()
    assert set(off) & per["xpu"]  # the blanket reaches intel again


def test_the_clause_stands_down_on_unmapped_devices(state, monkeypatch):
    """family_steps() is incomplete when a device cannot be placed, so
    nothing may subtract with it."""
    on = select(state, [CUDA_TU]).selected
    monkeypatch.setattr(state.preflight, "unmapped_devices", ["never-seen-device-9000"])
    stood_down = select(state, [CUDA_TU]).selected
    monkeypatch.setenv(ENV_VAR, "off")
    off = select(state, [CUDA_TU]).selected
    assert stood_down == off
    assert set(on) < set(stood_down)


def test_the_walker_never_maps_outside_csrc_and_cmake(state):
    """classify's docker/ case needs the unnarrowed answer, so the map must
    not own a docker/ path."""
    outside = [
        p for p in state.build_map.families if not p.startswith(("csrc/", "cmake/"))
    ]
    assert not outside, outside[:5]


def test_the_vocab_translation_is_pinned(state):
    """ "cuda" is the remainder carrying no device token, never a token
    family, or every main-image GPU suite would go. "other" is the token
    families minus amd and cpu."""
    per, union, nonfamily = state.family_partition()
    assert unions._build_map_allowed(state, frozenset({"cuda"})) == set(nonfamily)
    assert unions._build_map_allowed(state, frozenset({"amd"})) == set(per["amd"])
    assert (
        unions._build_map_allowed(state, frozenset({"other"}))
        == union - per["amd"] - per["cpu"]
    )
    both = unions._build_map_allowed(state, frozenset({"cuda", "cpu"}))
    assert both == set(nonfamily) | set(per["cpu"])


def test_a_mixed_unknown_known_chain_never_subtracts(tmp_path):
    """One unreadable arm makes the else-arm and any return() below it
    untrustworthy, so neither may narrow the rest of the file."""
    bm = BuildMap.build(
        _mini_repo(
            tmp_path,
            """
        if(SOME_UNKNOWN_FLAG)
          list(APPEND SRC "csrc/u.cu")
        elseif(VLLM_GPU_LANG STREQUAL "HIP")
          list(APPEND SRC "csrc/h.cu")
        else()
          return()
        endif()
        list(APPEND SRC "csrc/after.cu")
    """,
        )
    )
    assert "csrc/after.cu" not in bm.families  # every family still possible
    assert "csrc/u.cu" not in bm.families  # unknown arm inherits ALL
    assert bm.families["csrc/h.cu"] == frozenset({"amd"})  # own guard still counts
