import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        weights: str = "DEFAULT",
        output_shape: list = [68, 2],
    ):
        super().__init__()

        self.output_shape = output_shape
        backbone = models.get_model(name=model_name, weights=weights)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # freeze all weights
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

        num_features = backbone.fc.in_features
        self.output_layer = nn.Linear(
            in_features=num_features, out_features=output_shape[0] * output_shape[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        x = x.reshape(x.size(0), self.output_shape[0], self.output_shape[1])
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
