#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

source bin/activate
pip3 install --upgrade \
  ipympl \
  jupyter jupyterlab \
  jupyterlab-nvdashboard \
  matplotlib Pillow tensorflow
pip3 freeze > requirements.txt
