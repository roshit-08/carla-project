

source carla_env_37/bin/activate

kill previous servers 
```bash
killall -9 CarlaUE4-Linux-Shipping
```


terminal 1
```bash
cd /home/nikhil/Downloads/CARLA_0.9.13
./CarlaUE4.sh -quality-level=Low
```

terminal 2
```bash
cd /home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/examples
python3 manual_control.py
```

terminal 3
```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg && python3 src/run_carla_realtime.py --model-path artifacts/harsh_event_rf.joblib --mag-mode virtual3d --mag-field-strength 100 --mag-declination-deg 0 --mag-inclination-deg 60 --heuristic --print-safe --min-confidence 0.20 --consecutive-hits 1 --sensor-tick 0.02
```

---

### CNN Termional 3

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg && python3 src/run_carla_realtime_cnn.py \
  --model-path artifacts/harsh_event_cnn_bundle.pth \
  --mag-mode virtual3d \
  --mag-field-strength 100 \
  --mag-declination-deg 0 \
  --mag-inclination-deg 60 \
  --heuristic \
  --print-safe \
  --min-confidence 0.35 \
  --consecutive-hits 1 \
  --turn-threshold 20 \
  --sensor-tick 0.05
```



## XGBoost Terminal 3

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_realtime.py \
  --model-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --heuristic \
  --print-safe \
  --min-confidence 0.40 \
  --consecutive-hits 2 \
  --sensor-tick 0.05
```

## Random Forest v2 Terminal 3

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_realtime.py \
  --model-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --heuristic \
  --print-safe \
  --min-confidence 0.40 \
  --consecutive-hits 2 \
  --sensor-tick 0.05
```

## Multi-Model Ensemble + Live Driver Safety Score Terminal 3 - Manual drive   -  good

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_realtime_safety_score.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --rf-weight 0.40 \
  --xgb-weight 0.30 \
  --cnn-weight 0.30 \
  --min-confidence 0.40 \
  --consecutive-hits 2 \
  --sensor-tick 0.08
```
## Automated Scenario Benchmark (Autopilot + Scripted Harsh Events)

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_automated_scenario_old.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --min-confidence 0.40 \
  --consecutive-hits 2 \
  --recovery-rate 0.70 \
  --scenario-interval 20.0 \
  --sensor-tick 0.08 \
  --tm-port 8000 \
  --repeat-events 1

# latest (Auto-Recovery Enabled)
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
python3 src/run_carla_automated_scenario_old.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --rf-weight 0.55 \
  --xgb-weight 0.45 \
  --cnn-weight 0.00 \
  --min-confidence 0.50 \
  --consecutive-hits 2 \
  --recovery-rate 0.70 \
  --scenario-interval 20.0 \
  --sensor-tick 0.05 \
  --tm-port 8000 \
  --repeat-events 1


## Just Lane change auto driving

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_lane_change_scenario.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --rf-weight 0.55 \
  --xgb-weight 0.45 \
  --cnn-weight 0.00 \
  --min-confidence 0.50 \
  --consecutive-hits 2 \
  --recovery-rate 0.70 \
  --scenario-interval 20.0 \
  --sensor-tick 0.05 \
  --tm-port 8000 \
  --repeat-events 1
```


## terminal 3  - without manual.py - have gitters

export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_automated_scenario.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --min-confidence 0.60 \
  --consecutive-hits 1 \
  --recovery-rate 0.70 \
  --scenario-interval 20.0 \
  --sensor-tick 0.05 \
  --tm-port 8000 \
  --print-safe \
  --repeat-events 1


```

## Interactive Autopilot (Smooth Autopilot Driving + Manual Keyboard Event Injection)

```bash
export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg

python3 src/run_carla_autopilot_interactive.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --min-confidence 0.40 \
  --consecutive-hits 2 \
  --recovery-rate 0.30 \
  --print-safe \
  --sensor-tick 0.05



  ## New port issue
  export PYTHONPATH=$PYTHONPATH:/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
  python3 src/run_carla_autopilot_interactive.py \
  --rf-path notebooks/artifacts/harsh_event_rf_v2.joblib \
  --xgb-path notebooks/artifacts/harsh_event_xgb_v2.joblib \
  --cnn-path artifacts/harsh_event_cnn_bundle.pth \
  --min-confidence 0.40 \
  --consecutive-hits 1 \
  --recovery-rate 0.30 \
  --sensor-tick 0.05 \
  --tm-port 8010

```