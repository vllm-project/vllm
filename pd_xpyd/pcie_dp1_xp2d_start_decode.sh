BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

if [ -z "$1" ]; then
    echo "please input the tp size"
    echo "run with default mode n=1"
    TP_SIZE=1
else
    TP_SIZE=$1
fi

source "$BASH_DIR"/dp_start_decode.sh pcie2 16 $TP_SIZE 1 "10.112.110.161"
