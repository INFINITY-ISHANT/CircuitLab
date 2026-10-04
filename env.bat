@echo off
set PATH=%PATH%;C:\Program Files\Git\usr\bin
set PYTHONUTF8=1
rem Benchmark runs: hide the IOI paper and the ACDC paper from the literature agent. Remove this line for the discovery run.
set CIRCUITLAB_BLOCK_ARXIV=2211.00593,2304.14997
set PYTHONPATH=%PYTHONPATH%;%~dp0
