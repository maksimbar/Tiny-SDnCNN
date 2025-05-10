import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class SDnCNN(nn.Module):
    def __init__(
        self, num_layers=17, num_channels=64, activation="relu", dilation_rates=None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.dilation_rates = (
            dilation_rates if dilation_rates and len(dilation_rates) > 0 else [1]
        )

        logger.info(
            f"Initializing SDnCNN: layers={num_layers}, channels={num_channels}, activation={activation}, "
            f"dilation_rates (for intermediate layers, cyclic): {self.dilation_rates}"
        )

        # Store the class and its parameters
        self.activation_class = None
        self.activation_params = {}

        if activation.lower() == "relu":
            self.activation_class = nn.ReLU
            self.activation_params = {"inplace": True}
        elif activation.lower() == "leaky_relu":
            self.activation_class = nn.LeakyReLU
            self.activation_params = {"negative_slope": 0.1, "inplace": True}
        elif activation.lower() == "prelu":
            self.activation_class = nn.PReLU
            self.activation_params = {"num_parameters": num_channels}
        else:
            logger.warning(f"Unknown activation {activation}, defaulting to ReLU.")
            self.activation_class = nn.ReLU
            self.activation_params = {"inplace": True}

        layers = []
        # First layer
        layers.append(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1, bias=True, dilation=1)
        )
        layers.append(
            self.activation_class(**self.activation_params)
        )  # Correct instantiation

        # Intermediate layers
        num_intermediate_layers = num_layers - 2
        for i in range(num_intermediate_layers):
            current_dilation = self.dilation_rates[i % len(self.dilation_rates)]
            current_padding = current_dilation  # Assuming kernel_size=3, padding=dilation maintains size

            layers.append(
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    kernel_size=3,
                    padding=current_padding,
                    bias=False,
                    dilation=current_dilation,
                )
            )
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(
                self.activation_class(**self.activation_params)
            )  # Correct instantiation

        # Last layer
        layers.append(
            nn.Conv2d(num_channels, 1, kernel_size=3, padding=1, bias=True, dilation=1)
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
