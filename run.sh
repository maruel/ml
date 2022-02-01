#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

source bin/activate
# --watch ?
# --autoreload ?
jupyter lab -y --no-browser --ip 0.0.0.0 --notebook-dir notebooks
