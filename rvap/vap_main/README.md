## VAP to ONNX to TensorFlow to TensorFlow.js

1. Export ONNX

```bash
python vap_offline_onnx.py \
--vap_model ../../asset/vap/vap_state_dict_jp_20hz_2500msec.pt
```

2. ONNX to TensorFlow

```bash
pip install -U \
onnx==1.16.1 \
nvidia-pyindex \
onnx-graphsurgeon \
onnxruntime==1.18.1 \
onnxsim==0.4.33 \
simple_onnx_processing_tools \
sne4onnx>=1.0.13 \
sng4onnx>=1.0.4 \
tensorflow==2.17.0 \
protobuf==3.20.3 \
onnx2tf \
h5py==3.11.0 \
psutil==5.9.5 \
ml_dtypes==0.3.2 \
tf-keras~=2.16 \
flatbuffers>=23.5.26 \
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

![image](https://github.com/user-attachments/assets/d8c180c4-cc8e-42d9-b300-bfa84ee34155)
