import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, model_name: str = "resnet18", weights: str = "DEFAULT"):
        super().__init__()
        self.network = models.get_model(name="resnet18", weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, 136)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = x.reshape(x.size(0), 68, 2)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    resnet = ResNet()

    summary(
        model=resnet,
        input_size=[16, 3, 256, 256],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    # test input & output shape
    random_input = torch.randn([16, 3, 256, 256])
    output = resnet(random_input)
    print(f"\nINPUT SHAPE: {random_input.shape}")
    print(f"OUTPUT SHAPE: {output.shape}")
