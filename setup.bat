@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.
setlocal enableextensions

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

:: TODO: Check if it exists, and ask the user to run as an admin. The rest doesn't need elevated access.
:: https://pip.pypa.io/warnings/enable-long-paths
::powershell -Command "New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
::echo "This will require a reboot"

uv sync --group lint
