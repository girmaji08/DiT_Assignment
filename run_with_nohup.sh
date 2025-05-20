#!/bin/bash

# Check for correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_with_nohup.sh <script_to_run> <output_filename>"
    exit 1
fi

COMMAND="$1"
LOGFILE="$2"

# Run the command with nohup in foreground (sequential) and log output
nohup $COMMAND > "$LOGFILE" 2>&1

# # LOGFILE="${@: -1}"      # last argument is logfile
# # CMD=("${@:1:$#-1}")     # command and arguments (all but last)

# # nohup "${CMD[@]}" > "$LOGFILE" 2>&1


# # # usage: ./run_with_nohup.sh <command> <output_log_file>
# # COMMAND="$1"
# # shift
# # ARGS=("$@")
# # LOGFILE="${ARGS[-1]}"
# # unset 'ARGS[-1]'

# # nohup "$COMMAND" "${ARGS[@]}" > "$LOGFILE" 2>&1 &

# if [ $# -ne 2 ]; then
#     echo "Usage: $0 <script_to_run> <output_filename>"
#     exit 1
# fi

# SCRIPT_TO_RUN=$1
# OUTPUT_FILE=$2

# nohup "$SCRIPT_TO_RUN" > "$OUTPUT_FILE" 2>&1 &
