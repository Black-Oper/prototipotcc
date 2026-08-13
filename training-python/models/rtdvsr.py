import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .blocks import HAS_DEFORM_CONV, ResidualBlock, ConvGRU, DeformableAligner, ConvAligner

# ---------------------------------------------------------------------------
# Arquitetura: RTDVSR — Real-Time Deformable Video Super-Resolution
#
# Fundamentação:
#   - Shi et al., 2016 (ESPCN)
#       Extração de features inteiramente no espaço LR + sub-pixel shuffle.
#       Reduz complexidade computacional e de memória em ordens de magnitude.
#
#   - Fuoli et al., 2023 (DAP)
#       Arquitetura recorrente estritamente causal com alinhamento deformável.
#       Permite inferência online (streaming) sem depender de frames futuros.
#       DAP-128 atinge >26 FPS com qualidade competitiva.
#
#   - Miyazaki et al., 2024 (RSRDCN)
#       Convolução deformável para alinhamento de features, substituindo
#       optical flow e patch-matching. 30x mais rápido que SRNTT.
#
#   - Caballero et al., 2017
#       Fusão espaço-temporal no espaço LR. 3-5 frames é o ponto ótimo;
#       acima disso, ruído temporal degrada o desempenho.
#
#   - Huang et al., 2025 (LightVSR)
#       Design leve com ~3.5M params atingindo ~28 FPS.
#       Blocos residuais e atenção simplificada (SE-attention).
#
#   - Wang et al., 2019 / Xi et al., 2025 (Surveys)
#       Pós-amostragem (post-upsampling) é a estratégia mais eficiente.
#       Redes leves são o futuro para equilíbrio qualidade/velocidade.
#
#   - Wang et al., 2025 (HarmoQ / Outlier-Aware PTQ)
#       Quantização pós-treinamento pode acelerar até 75x. Arquitetura
#       deve ser "quantization-friendly" (evitar outliers extremos).
# ---------------------------------------------------------------------------


@register_model("RTDVSR", interface="recurrent")
class RTDVSR(nn.Module):
    """
    Real-Time Deformable Video Super-Resolution.

    Arquitetura recorrente causal com alinhamento implícito via convolução
    deformável e fusão temporal via ConvGRU. Todo o processamento ocorre
    no espaço de baixa resolução até o estágio final de sub-pixel shuffle.

    Pipeline (todo no espaço LR):
        Frame LR ──► Extração de Features ──► Alinhamento Deformável ──►
        ConvGRU (fusão temporal) ──► Refinamento Residual+SE ──►
        Sub-pixel Shuffle + Skip Bicúbico ──► Frame SR

    O estado oculto (hidden state) carrega informação temporal dos frames
    anteriores, permitindo inferência causal (online) sem buffering.

    Args:
        scale_factor: Fator de upscaling (padrão: 2)
        channels: Canais da imagem (padrão: 3 para RGB)
        hidden_dim: Dimensão dos feature maps internos (padrão: 64)
        num_res_blocks: Quantidade de blocos residuais de refinamento (padrão: 4)
    """

    def __init__(self, scale_factor=2, channels=3, hidden_dim=64, num_res_blocks=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        self.feat_extract = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 5, padding=2),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.PReLU(),
        )

        if HAS_DEFORM_CONV:
            self.aligner = DeformableAligner(hidden_dim)
        else:
            print("[AVISO] torchvision.ops.DeformConv2d não disponível. "
                  "Usando alinhamento por convolução padrão (qualidade reduzida).")
            self.aligner = ConvAligner(hidden_dim)

        self.fusion = ConvGRU(hidden_dim)

        self.refine = nn.Sequential(
            *[ResidualBlock(hidden_dim, use_attention=(i % 2 == 1))
              for i in range(num_res_blocks)]
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_dim, channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self._initialize_weights()

    def forward(self, x, prev_state=None):
        """
        Args:
            x: Frame LR atual (B, C, H, W)
            prev_state: Estado oculto do frame anterior (B, hidden_dim, H, W) ou None

        Returns:
            sr: Frame SR reconstruído (B, C, H*scale, W*scale)
            state: Estado oculto atualizado para o próximo frame
        """
        feat = self.feat_extract(x)

        if prev_state is None:
            prev_state = torch.zeros_like(feat)

        aligned = self.aligner(feat, prev_state)
        state = self.fusion(feat, aligned)
        refined = self.refine(state) + feat

        residual = self.upsample(refined)
        base = F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bicubic', align_corners=False)
        sr = residual + base

        return sr, state

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
