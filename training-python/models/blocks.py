import torch
import torch.nn as nn

try:
    from torchvision.ops import DeformConv2d
    HAS_DEFORM_CONV = True
except ImportError:
    HAS_DEFORM_CONV = False


class SEBlock(nn.Module):
    """Squeeze-and-Excitation para atenção por canal (Huang et al., 2025).

    Recalibra os feature maps por canal, priorizando canais com
    informação de textura e movimento mais relevantes.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x))


class ResidualBlock(nn.Module):
    """Bloco residual leve com SE-attention opcional."""

    def __init__(self, channels, use_attention=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.attention = SEBlock(channels) if use_attention else nn.Identity()

    def forward(self, x):
        return x + self.attention(self.body(x))


class ConvGRU(nn.Module):
    """Unidade recorrente convolucional para fusão temporal (Fuoli et al., 2023).

    Mantém um estado oculto que acumula informação temporal de forma
    controlada via gates de reset e update, garantindo coerência
    temporal em sequências longas sem degradação de qualidade.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_reset = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_update = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv_candidate = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)

    def forward(self, feat, prev_state):
        combined = torch.cat([feat, prev_state], dim=1)
        reset = torch.sigmoid(self.conv_reset(combined))
        update = torch.sigmoid(self.conv_update(combined))
        candidate = torch.tanh(
            self.conv_candidate(torch.cat([feat, reset * prev_state], dim=1))
        )
        return (1 - update) * prev_state + update * candidate


class DeformableAligner(nn.Module):
    """Alinhamento temporal via convolução deformável (Fuoli 2023, Miyazaki 2024).

    Prediz offsets a partir da concatenação [features_atuais, estado_anterior]
    e aplica convolução deformável no estado anterior para alinhá-lo ao frame
    corrente. Substitui optical flow com custo muito menor.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, 2 * 3 * 3, 3, padding=1),  # 18 offsets para kernel 3x3
        )
        self.deform_conv = DeformConv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, feat, prev_state):
        offsets = self.offset_conv(torch.cat([feat, prev_state], dim=1))
        return self.deform_conv(prev_state, offsets)


class ConvAligner(nn.Module):
    """Fallback: alinhamento via convolução padrão quando DeformConv2d não
    está disponível. Menos eficaz, mas funcional."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, feat, prev_state):
        return self.align(torch.cat([feat, prev_state], dim=1))
