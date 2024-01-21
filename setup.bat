@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.

if not exist venv\Script\activate.bat python -m venv venv
call upgrade.bat
