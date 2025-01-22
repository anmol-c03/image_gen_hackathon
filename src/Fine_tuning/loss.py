import torch
'''
TODO:
1. Define Pattern loss
2. Define style loss
3. Define Geometric loss
'''
class loss_fn:
    def __init__(self,original,generated):
        self.original = original
        self.generated = generated
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.feature_layers = ['3', '15', '25']  # to extract low, mid and high level features using VGG16
        self.weights={
            'pattern':1,
            "style":1,
            'geometric':1
        }

    def forward(self):
        pattern_loss= self.pattern_loss()
        style_loss = self.style_loss()
        geometric_loss = self.geometric_loss()
        return (self.weights['pattern']*pattern_loss+
                self.weights['style']*style_loss+
                self.weights['geometric']*geometric_loss)
        


    def extract_features(self,x):
        features = []
        for name, module in self.model.features._modules.items():
            feature = module(x)
            if name in self.feature_layers:
                features.append(feature)
        return features
    def pattern_loss(self):
        loss_p=torch.nn.MSELoss()(
            self.extract_features(self.original),
            self.extract_features(self.generated)
        )
        return loss_p.sum()

    def geometric_loss(self):
        pass

    def style_loss(self):
        pass
