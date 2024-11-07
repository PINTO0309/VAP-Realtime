import os
import onnx
from onnxsim import simplify
from spo4onnx import partial_optimization
import torch
import soundfile as sf
from vap_main import VAPRealTime, AudioEncoder
import argparse
from typing import List

if __name__ == "__main__":

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vap_model", type=str, default='../../asset/vap/vap_state_dict_jp_20hz_2500msec.pt')
    parser.add_argument("--cpc_model", type=str, default='../../asset/cpc/60k_epoch4-d0f474de.pt')
    parser.add_argument("--filename_output", type=str, default='output_offline.txt')
    parser.add_argument("--input_wav_left", type=str, default='../../input/wav_sample/jpn_inoue_16k.wav')
    parser.add_argument("--input_wav_right", type=str, default='../../input/wav_sample/jpn_sumida_16k.wav')
    parser.add_argument("--vap_process_rate", type=int, default=20)
    parser.add_argument("--context_len_sec", type=float, default=5)
    parser.add_argument("--gpu", action='store_true')
    args = parser.parse_args()

    #
    # GPU Usage
    #
    device = torch.device('cpu')
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
    print('Device: ', device)

    vap = AudioEncoder(args.vap_model, args.cpc_model, device, args.vap_process_rate, args.context_len_sec)

    basename_without_ext_vap_model = os.path.splitext(os.path.basename(args.vap_model))[0]

    vap.eval()
    vap.cpu()


    # vap_state_dict_jp_20hz_2500msec
    #   x1_.shape
    #       torch.Size([1, 1, 1120])
    #   x2_.shape
    #       torch.Size([1, 1, 1120])
    onnx_file = f"{basename_without_ext_vap_model}.onnx"
    x1 = torch.randn(1, 1, 1120)
    x2 = torch.randn(1, 1, 1120)
    torch.onnx.export(
        vap,
        args=(x1,x2),
        f=onnx_file,
        opset_version=14,
        input_names=['input_1','input_2'],
        output_names=['p_now', 'p_future'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    # Einsum optimization
    partial_optimization(
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=onnx_file,
    )

    model = onnx.load(onnx_file)
    input_shape: List[str] = [str(dim.dim_value) for dim in model.graph.input[0].type.tensor_type.shape.dim]
    rename_file = f"{os.path.splitext(os.path.basename(onnx_file))[0]}_{'x'.join(input_shape)}.onnx"
    os.rename(onnx_file, rename_file)