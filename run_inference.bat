@echo off
echo Starting Real-Time Inference Demo...
echo.
echo This script effectively simulates a real-time feed by processing a sample pair.
echo.
.venv\Scripts\python.exe src/inference.py --pre_path "data/SN8/Germany_Training_Public/PRE-event/10500500C4DD7000_0_15_63.tif"
echo.
pause
