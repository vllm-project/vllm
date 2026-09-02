# Downloaded ci-infra

What `pytest tests --sync` last pulled from `vllm-project/ci-infra`, which holds
the pipeline generator. `ci_selector` reproduces parts of it by hand, and this
is the only thing that can say whether those copies are still right.

`values.json` is every string constant the generator defines, evaluated. The
offline tests read it and compare against `handwritten.py`. Extraction is
deliberately generic rather than limited to the values we watch, so watching
one more is a new test and not another download.

The `.py.txt` files are functions we transcribe or read a literal out of,
reparsed and reprinted with docstrings dropped, so formatting and comments do
not register as changes. They are data, not code: the extension keeps ruff and
mypy off them, since they name things from modules this repo does not have.

Four of the `.py.txt` files are **executed** by the tests and run against our
versions on generated inputs, which is what makes those checks self-clearing:
there is no approval to store and no baseline to maintain.

`manifest.json` records the upstream commit and the file list, nothing else.
That commit is the ref the files were fetched at, and it is the last one to
touch the generator rather than the tip of ci-infra's default branch, which
moves for unrelated reasons. Fetching that sha reproduces this snapshot.

Two of the functions cannot be executed: `select_steps_and_dependencies`, which
we depend on rather than reproduce, and `read_steps_from_job_dir`, which is here
only for the working-dir default it assigns. Neither has a test that can fail,
so those two are kept tracked and a sync that changes either shows up as a git
diff to read. That is weaker than a test and is the honest residual.

Do not hand-edit. To refresh:

    uv run pytest tests --sync -q     # download, then run the checks
    uv run python tests/ci_infra.py   # download only

Everything here is gitignored and rebuilt by `--sync`, except the two
unexecutable functions named above. Until you sync once the offline checks skip,
and `test_the_snapshot_is_armed` fails so an unarmed suite is never green.
