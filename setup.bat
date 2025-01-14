@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.
setlocal enableextensions

:: TODO: Check if it exists, and ask the user to run as an admin. The rest doesn't need elevated access.
:: https://pip.pypa.io/warnings/enable-long-paths
::powershell -Command "New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
::echo "This will require a reboot"

if not exist venv\Script\activate.bat python -m venv venv
call upgrade.bat
