BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

if [ -z "$1" ]; then
    echo "please input the tp size"
    echo "run with default mode n=1"
    TP_SIZE=1
else
    TP_SIZE=$1
fi

source "$BASH_DIR"/dp_start_decode.sh g13 32 $TP_SIZE 0 "10.239.129.81"

