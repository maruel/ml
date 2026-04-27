#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

UNAME=$(uname)

PKG_GROUPS="--group lint"

if [ "$UNAME" != "Darwin" ]; then
  if lspci | grep -i intel > /dev/null 2>&1; then
    PKG_GROUPS="$PKG_GROUPS --group intel"
  fi
  if lspci | grep -i nvidia > /dev/null 2>&1; then
    PKG_GROUPS="$PKG_GROUPS --group cuda"
  fi
fi

echo "Upgrading all dependencies"
uv lock --upgrade
uv sync $PKG_GROUPS
