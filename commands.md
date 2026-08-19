source the virtual env : carla_env37

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

### Running with the 1D-CNN Model

Terminal 3 (CNN Real-time Detector):
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

> **Note:** After retraining the CNN notebook (`notebooks/harsh_event_detection_cnn.ipynb`),
> the model file at `artifacts/harsh_event_cnn_bundle.pth` will include the new
> sustained-yaw features and be trained with turns in the test set.
> Use `--consecutive-hits 2` if you see too many false positives in normal driving.