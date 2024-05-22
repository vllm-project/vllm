#!/usr/bin/env bash

set -e
set -u

usage() {
    echo``
    echo "Archive the input benchmark-data folder/file at the destination-root, under"
    echo "a well-defined directory structure."
    echo "the directory structure is as follows,"
    echo " <destination-root>/date/github_event/label/branch/commit-hash/github_run_id/"
    echo "The script makes th"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -i    - path to the folder/file to archive"
    echo "  -o    - path to the destination-root"
    echo "  -e    - github event name"
    echo "  -l    - github instance label"
    echo "  -p    - python version"
    echo "  -b    - github branch name"
    echo "  -c    - github commit hash"
    echo "  -r    - github run id"
    echo
}

# Empty strings are invalid
INPUT_PATH=""
OUTPUT_PATH=""
GITHUB_EVENT_NAME=""
GITHUB_LABEL=""
PYTHON_VERSION=""
GITHUB_BRANCH=""
GITHUB_COMMIT=""
GITHUB_RUN_ID=""

while getopts "hi:o:e:l:p:b:c:r:" OPT; do
    case "${OPT}" in
        h)
            usage
            exit 1
            ;;
        i)
            INPUT_PATH="${OPTARG}"
            ;;
        o)
            OUTPUT_PATH="${OPTARG}"
            ;;
        e)
            GITHUB_EVENT_NAME="${OPTARG}"
            ;;
        l)
            GITHUB_LABEL="${OPTARG}"
            ;;
        p)
            PYTHON_VERSION="${OPTARG}"
            ;;
        b)
            GITHUB_BRANCH="${OPTARG}"
            ;;
        c)
            GITHUB_COMMIT="${OPTARG}"
            ;;
        r)
            GITHUB_RUN_ID="${OPTARG}"
            ;;
    esac
done

# logging
echo "Args :"
echo "INPUT_PATH : ${INPUT_PATH}"
echo "OUTPUT_PATH : ${OUTPUT_PATH}"
echo "GITHUB_EVENT_NAME : ${GITHUB_EVENT_NAME}"
echo "GITHUB_LABEL : ${GITHUB_LABEL}"
echo "PYTHON VERSION: ${PYTHON_VERSION}"
echo "GITHUB_BRANCH : ${GITHUB_BRANCH}"
echo "GITHUB_COMMIT : ${GITHUB_COMMIT}"
echo "GITHUB_RUN_ID : ${GITHUB_RUN_ID}"

# Make sure we have all the information to construct a correct path
if [[ "${INPUT_PATH}" == "" || "${OUTPUT_PATH}" == "" || "${GITHUB_EVENT_NAME}" == "" || "${GITHUB_LABEL}" == "" || "${PYTHON_VERSION}" == "" || "${GITHUB_BRANCH}" == "" || "${GITHUB_COMMIT}" == "" || "${GITHUB_RUN_ID}" == "" ]];
then
  echo "Error : Incomplete arg list - Atleast one of the arguments is an empty string"
  exit 1
fi

# Branch names can have '/' - replace with '_'
GITHUB_BRANCH=`echo ${GITHUB_BRANCH} | tr "/" "_"`
# Using the full commit hash is a over-kill
GITHUB_COMMIT=${GITHUB_COMMIT:0:7}
# Get today's date
TODAY=`date '+%Y-%m-%d'`

DESTINATION_DIR=${OUTPUT_PATH}/${TODAY}/${GITHUB_EVENT_NAME}/${GITHUB_LABEL}/${PYTHON_VERSION}/${GITHUB_BRANCH}/${GITHUB_COMMIT}/${GITHUB_RUN_ID}
echo "Destination DIR : ${DESTINATION_DIR}"

# Create destination dir
mkdir -p ${DESTINATION_DIR}

INPUT_BASENAME=`basename ${INPUT_PATH}`
DESTINATION_TAR=${DESTINATION_DIR}/${INPUT_BASENAME}.tar
echo "Destination TAR : ${DESTINATION_TAR}"
# Fail if destination tar exists already
if [[ -f ${DESTINATION_TAR} ]];
then
  echo "Path ${DESTINATION_TAR} exists already"
  exit 1
fi
# Tar file
tar -cvf ${DESTINATION_TAR} ${INPUT_PATH}

exit 0
