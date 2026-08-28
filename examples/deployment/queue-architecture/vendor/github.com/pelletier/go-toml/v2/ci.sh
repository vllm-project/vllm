#!/usr/bin/env bash


stderr() {
    echo "$@" 1>&2
}

usage() {
    b=$(basename "$0")
    echo $b: ERROR: "$@" 1>&2

    cat 1>&2 <<EOF

DESCRIPTION

    $(basename "$0") is the script to run continuous integration commands for
    go-toml on unix.

    Requires Go and Git to be available in the PATH. Expects to be ran from the
    root of go-toml's Git repository.

USAGE

    $b COMMAND [OPTIONS...]

COMMANDS

benchmark [OPTIONS...] [BRANCH]

    Run benchmarks.

    ARGUMENTS

        BRANCH Optional. Defines which Git branch to use when running
               benchmarks.

    OPTIONS

        -d      Compare benchmarks of HEAD with BRANCH using benchstats. In
                this form the BRANCH argument is required.

        -a      Compare benchmarks of HEAD against go-toml v1 and
                BurntSushi/toml.

        -html   When used with -a, emits the output as HTML, ready to be
                embedded in the README.

coverage [OPTIONS...] [BRANCH]

    Generates code coverage.

    ARGUMENTS

        BRANCH  Optional. Defines which Git branch to use when reporting
                coverage. Defaults to HEAD.

    OPTIONS

        -d      Compare coverage of HEAD with the one of BRANCH. In this form,
                the BRANCH argument is required. Exit code is non-zero when
                coverage percentage decreased.
EOF
    exit 1
}

cover() {
    branch="${1}"
    dir="$(mktemp -d)"

    stderr "Executing coverage for ${branch} at ${dir}"

    if [ "${branch}" = "HEAD" ]; then
	    cp -r . "${dir}/"
    else
	    git worktree add "$dir" "$branch"
    fi

    pushd "$dir"
    go test -covermode=atomic  -coverpkg=./... -coverprofile=coverage.out.tmp ./...
    grep -Ev '(fuzz|testsuite|tomltestgen|gotoml-test-decoder|gotoml-test-encoder)' coverage.out.tmp > coverage.out
    go tool cover -func=coverage.out
    echo "Coverage profile for ${branch}: ${dir}/coverage.out" >&2
    popd

    if [ "${branch}" != "HEAD" ]; then
	    git worktree remove --force "$dir"
    fi
}

coverage() {
    case "$1" in
	-d)
	    shift
	    target="${1?Need to provide a target branch argument}"

	    output_dir="$(mktemp -d)"
	    target_out="${output_dir}/target.txt"
	    head_out="${output_dir}/head.txt"
	    
	    cover "${target}" > "${target_out}"
	    cover "HEAD" > "${head_out}"

	    cat "${target_out}"
	    cat "${head_out}"

	    echo ""

	    target_pct="$(tail -n2 ${target_out} | head -n1 | sed -E 's/.*total.*\t([0-9.]+)%.*/\1/')"
	    head_pct="$(tail -n2 ${head_out} | head -n1 | sed -E 's/.*total.*\t([0-9.]+)%/\1/')"
	    echo "Results: ${target} ${target_pct}% HEAD ${head_pct}%"

	    delta_pct=$(echo "$head_pct - $target_pct" | bc -l)
	    echo "Delta: ${delta_pct}"

	    if [[ $delta_pct = \-* ]]; then
		    echo "Regression!";

            target_diff="${output_dir}/target.diff.txt"
            head_diff="${output_dir}/head.diff.txt"
            cat "${target_out}" | grep -E '^github.com/pelletier/go-toml' | tr -s "\t " | cut -f 2,3 | sort > "${target_diff}"
            cat "${head_out}" | grep -E '^github.com/pelletier/go-toml' | tr -s "\t " | cut -f 2,3 | sort > "${head_diff}"

            diff --side-by-side --suppress-common-lines "${target_diff}" "${head_diff}"
		    return 1
	    fi
	    return 0
	    ;;
    esac

    cover "${1-HEAD}"
}

bench() {
    branch="${1}"
    out="${2}"
    replace="${3}"
    dir="$(mktemp -d)"

    stderr "Executing benchmark for ${branch} at ${dir}"

    if [ "${branch}" = "HEAD" ]; then
    	cp -r . "${dir}/"
    else
	    git worktree add "$dir" "$branch"
    fi

    pushd "$dir"

    if [ "${replace}" != "" ]; then
        find ./benchmark/ -iname '*.go' -exec sed -i -E "s|github.com/pelletier/go-toml/v2|${replace}|g" {} \;
        go get "${replace}"
    fi

    export GOMAXPROCS=2
    go test '-bench=^Benchmark(Un)?[mM]arshal' -count=10 -run=Nothing ./... | tee "${out}"
    popd

    if [ "${branch}" != "HEAD" ]; then
	    git worktree remove --force "$dir"
    fi
}

fmktemp() {
    if mktemp --version &> /dev/null; then
	# GNU
        mktemp --suffix=-$1
    else
	# BSD
	mktemp -t $1
    fi
}

benchstathtml() {
python3 - $1 <<'EOF'
import sys

lines = []
stop = False

with open(sys.argv[1]) as f:
    for line in f.readlines():
        line = line.strip()
        if line == "":
            stop = True
        if not stop:
            lines.append(line.split(','))

results = []
for line in reversed(lines[2:]):
    if len(line) < 8 or line[0] == "":
        continue
    v2 = float(line[1])
    results.append([
        line[0].replace("-32", ""),
        "%.1fx" % (float(line[3])/v2),  # v1
        "%.1fx" % (float(line[7])/v2),  # bs
    ])
# move geomean to the end
results.append(results[0])
del results[0]


def printtable(data):
    print("""
<table>
    <thead>
        <tr><th>Benchmark</th><th>go-toml v1</th><th>BurntSushi/toml</th></tr>
    </thead>
    <tbody>""")

    for r in data:
        print("        <tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(*r))

    print("""     </tbody>
</table>""")


def match(x):
    return "ReferenceFile" in x[0] or "HugoFrontMatter" in x[0]

above = [x for x in results if match(x)]
below = [x for x in results if not match(x)]

printtable(above)
print("<details><summary>See more</summary>")
print("""<p>The table above has the results of the most common use-cases. The table below
contains the results of all benchmarks, including unrealistic ones. It is
provided for completeness.</p>""")
printtable(below)
print('<p>This table can be generated with <code>./ci.sh benchmark -a -html</code>.</p>')
print("</details>")

EOF
}

benchmark() {
    case "$1" in
    -d)
        shift
     	target="${1?Need to provide a target branch argument}"

        old=`fmktemp ${target}`
        bench "${target}" "${old}"

        new=`fmktemp HEAD`
        bench HEAD "${new}"

        benchstat "${old}" "${new}"
        return 0
        ;;
    -a)
        shift

        v2stats=`fmktemp go-toml-v2`
        bench HEAD "${v2stats}" "github.com/pelletier/go-toml/v2"
        v1stats=`fmktemp go-toml-v1`
        bench HEAD "${v1stats}" "github.com/pelletier/go-toml"
        bsstats=`fmktemp bs-toml`
        bench HEAD "${bsstats}" "github.com/BurntSushi/toml"

        cp "${v2stats}" go-toml-v2.txt
        cp "${v1stats}" go-toml-v1.txt
        cp "${bsstats}" bs-toml.txt

        if [ "$1" = "-html" ]; then
            tmpcsv=`fmktemp csv`
            benchstat -format csv go-toml-v2.txt go-toml-v1.txt bs-toml.txt > $tmpcsv
            benchstathtml $tmpcsv
        else
            benchstat go-toml-v2.txt go-toml-v1.txt bs-toml.txt
        fi

        rm -f go-toml-v2.txt go-toml-v1.txt bs-toml.txt
        return $?
    esac

    bench "${1-HEAD}" `mktemp`
}

case "$1" in
    coverage) shift; coverage $@;;
    benchmark) shift; benchmark $@;;
    *) usage "bad argument $1";;
esac
