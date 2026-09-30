#!/usr/bin/env bash
set -euo pipefail

if (( $# != 7 )); then
    echo "Usage: $0 XNAT_HOST XNAT_USER XNAT_PASS SESSION_ID RESOURCE_DIR FILE_EXT OUTPUT_DIR" >&2
    exit 2
fi

export XNAT_HOST=$1
export XNAT_USER=$2
export XNAT_PASS=$3
session_id=$4
resource_dir=$5
file_ext=$6
output_dir=$7

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$script_dir/download_xnat_resource.py" download-resource \
    "$session_id" "$resource_dir" "$file_ext" "$output_dir"
