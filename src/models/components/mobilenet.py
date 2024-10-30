import torch
import torch.nn as nn
from torchvision import models

class MobileNet(nn.Module):
    def __init__(
        self,
        model_name: str = 'mobilenet_v3_large',
        weights: str = 'MobileNet_V3_Large_Weights.IMAGENET1K_V2',
        output_shape: list = [68, 2],
        transfer: bool = True,
    ):
        super().__init__()
        self.output_shape = output_shape
        # Use pretrained model
        self.network = models.get_model(name=model_name, weights=weights)
        # Change classifier layer
        num_ftus = self.network.classifier[0].in_features
        self.network.classifier = nn.Sequential(
            nn.Linear(num_ftus, 512, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, self.output_shape[0] * self.output_shape[1], bias=True)
        )
        if transfer:
            self.transfer()
        else:
            self.finetune()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = x.reshape(x.size(0), self.output_shape[0], self.output_shape[1])
        return x

    def transfer(self):
        """Freeze weights for transfer learning"""
        for p in self.network.parameters():
            p.requires_grad = False
        for p in self.network.classifier.parameters():
            p.requires_grad = True

    def finetune(self):
        """Unfreeze weights to finetune"""
        for p in self.network.parameters():
            p.requires_grad = True

if __name__ == "__main__":
    from torchinfo import summary

    mobilenet = MobileNet(transfer=True)

    summary(
        model=mobilenet,
        input_size=[16, 3, 256, 256],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    # test input & output shape
    random_input = torch.randn([16, 3, 256, 256])
    output = mobilenet(random_input)
    print(f"\nINPUT SHAPE: {random_input.shape}")
    print(f"OUTPUT SHAPE: {output.shape}")