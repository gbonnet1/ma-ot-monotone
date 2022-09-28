#!/bin/sh

set -e

if test "$#" -ne 1
then
    echo "Usage: $0 path/to/appleseed.cli" >&2
    exit 1
fi

"$1" \
    -o "$(dirname "$0")/../figures/out/optics_simulation.png" \
    "$(dirname "$0")/simulation.appleseed"
