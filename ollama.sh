#!/bin/bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

cd "$(dirname $0)"

if ! which ollama > /dev/null; then
	go install github.com/ollama/ollama@latest
fi

if ollama --version | grep -q Warning; then
	echo "Run: ollama serve"
	exit 1
fi

ollama run huggingface.co/lmstudio-community/DeepCoder-14B-Preview-GGUF:Q4_K_M
