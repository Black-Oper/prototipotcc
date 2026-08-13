import sys
import os
from pathlib import Path

import torch
import torchvision
import torchvision.ops
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rtdvsr import RTDVSR

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Ponte de Tradução: TorchVision C++ -> ONNX Padrão (DeformConv - Opset 19)
# ---------------------------------------------------------------------------
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
def deform_conv2d_symbolic_opset19(
    g,
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask
):
    # ONNX DeformConv (Opset 19) exige a ordem de inputs: X, W, offset, B, mask
    # E o padding no formato de 4 inteiros [top, left, bottom, right]
    return g.op(
        "DeformConv",
        input,
        weight,
        offset,
        bias,
        mask,
        strides_i=[stride_h, stride_w],
        pads_i=[pad_h, pad_w, pad_h, pad_w],
        dilations_i=[dil_h, dil_w],
        group_i=n_weight_grps,
        offset_group_i=n_offset_grps
    )

register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic_opset19, 19)


def export_rtdvsr_to_onnx():
    # Configurações de dimensão (LR: 960x540 -> SR: 1920x1080)
    batch_size = 1
    channels = 3
    height_lr = 540
    width_lr = 960
    hidden_dim = 64
    scale_factor = 2

    checkpoint_path = "checkpoints/RTDVSR_best_model.pth"
    onnx_output_path = "../inference-cpp/rtdvsr.onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Export] Dispositivo detectado: {device}")

    # Instanciar o modelo original intacto do TCC (com DeformConv2d e 6 blocos)
    model = RTDVSR(
        scale_factor=scale_factor,
        channels=channels,
        hidden_dim=hidden_dim,
        num_res_blocks=6
    ).to(device)

    if os.path.exists(checkpoint_path):
        print(f"[Export] Carregando os pesos de: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"[AVISO] Checkpoint '{checkpoint_path}' não encontrado. Exportando com pesos inicializados.")

    model.eval()

    # Criar inputs sintéticos de entrada
    dummy_x = torch.randn(batch_size, channels, height_lr, width_lr, device=device, dtype=torch.float32)
    dummy_state = torch.zeros(batch_size, hidden_dim, height_lr, width_lr, device=device, dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)

    print("[Export] Exportando o modelo para formato ONNX (Opset 19 - DeformConv Oficial)...")
    
    # Exportação em Opset 19
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_x, dummy_state),
            onnx_output_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=['x', 'prev_state'],
            output_names=['sr', 'state']
        )

    print(f"[Sucesso] Grafo ONNX gerado com operador DeformConv padrão em: {onnx_output_path}")

if __name__ == "__main__":
    export_rtdvsr_to_onnx()