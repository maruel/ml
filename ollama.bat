@echo off
:: Copyright 2025 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.
setlocal enableextensions


:: if where ollama.exe > /dev/null; then
::  https://ollama.com/download
::  https://github.com/ollama/ollama/issues/9012
::	go install github.com/ollama/ollama@latest
:: fi

::if ollama --version | grep -q Warning; then
::	echo "Run: ollama serve"
::	exit 1
::fi

ollama.exe pull nomic-embed-text
ollama.exe run huggingface.co/lmstudio-community/DeepCoder-14B-Preview-GGUF:Q4_K_M
