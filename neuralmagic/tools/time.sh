#!/bin/bash

#
# A wrapper around a command that produces uniform time/memory usage results.  Works
# with /usr/bin/time or builtin bash time (linux)
#

OUTPUT=timing.log

usage() {
    echo "Usage: $0 <options> command-to-time"
    echo "  -o <file>          - output file, results will be appended to this file"
    exit 1
}

while getopts "o:h" OPT; do
    case "${OPT}" in
        o)
            OUTPUT="${OPTARG}"
            ;;
        h)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ $# -eq 0 ]; then
    echo "Nothing to time, exiting."
    exit 1
fi

# Simple heuristic (search for the -o argument) to determine the file name
# being compiled.  Since this records timing information for every compiler
# related command in the makefile, some times may be for dependency generation
# (.d) or object file creation (.cpp/.o) or linking (.bin)
# First look for a -MF FOO.d file as the output.  If there isn't one, then look
# for -o FOO.
ARGS="$@"
DEP_FILE=$(echo "${ARGS}" | sed -e "s/.*\-MF  *\([^\ ][^\ ]*[\.]d\).*/\1/")
if [ "${DEP_FILE}" = "${ARGS}" ]; then
    FILE=$(echo "${ARGS}" | sed -e "s/.*\-o  *\([^\ ][^\ ]*\).*/\1/")
else
    FILE="${DEP_FILE}"
fi

TIME_OPTIONS="[Timing] ${FILE}: elapsed=%e user=%U system=%S maxrss=%M avgrss=%t avgmem=%K avgdata=%D"

if [ -x /usr/bin/time ]; then
    /usr/bin/time -o "${OUTPUT}" -a -f "${TIME_OPTIONS}" "$@"
else
    BASH=$(command -v bash)
    FMT="[Timing] ${FILE}: elapsed=%R user=%U system=%S maxrss=0 avgrss=0 avgmem=0 avgdata=0"
    CMD=$(echo "$@" | sed -e s'/"/\\"/g' | sed -e s'/\$/\\$/g')
    OUT_TMP=$(${BASH} -c "export TIMEFORMAT=\"$FMT\"; time $CMD" 2>&1)
    STATUS=$?
    echo "${OUT_TMP}" | grep -F "[Timing] ${FILE}:" >> "${OUTPUT}"
    echo "${OUT_TMP}" | grep -Fv "[Timing] ${FILE}:" || true
    exit ${STATUS}
fi
