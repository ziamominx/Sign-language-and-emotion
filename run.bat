@echo off
rem Zia's Glasses launcher — uses Python 3.10 (required for mediapipe 0.10.14)
rem PYTHONUTF8=1 prevents a deepface Unicode crash on the Windows cp1252 console
set PYTHONUTF8=1
set TF_CPP_MIN_LOG_LEVEL=3
set GLOG_minloglevel=3
py -3.10 zias_glasses.py
