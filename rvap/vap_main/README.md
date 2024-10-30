## VAP to ONNX to TensorFlow to TensorFlow.js

1. Export ONNX

```bash
python vap_offline_onnx.py \
--vap_model ../../asset/vap/vap_state_dict_jp_20hz_2500msec.pt
```

2. ONNX to TensorFlow

```bash
pip install -U \
onnx2tf \
spo4onnx

pip install -U --no-deps \
tensorflowjs \
tensorflow_decision_forests \
ydf \
tensorflow_hub

onnx2tf \
-i vap_realtime_jp_20hz_2500msec_1x1x1120.onnx \
-kat input_1 input_2 \
-cotof \
-coion
```
![image](https://github.com/user-attachments/assets/b634971b-5bf0-4b1f-8a65-c4194148fdc1)


3. TensorFlow to TensorFlow.js

```bash
tensorflowjs_converter \
--input_format tf_saved_model \
--output_format tfjs_graph_model \
saved_model \
tfjs_model
```
![image](https://github.com/user-attachments/assets/4a009bde-3464-4dd6-a5de-8da3099d6f71)
