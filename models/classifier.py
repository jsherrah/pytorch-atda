"""Feature Classifier for ATDA.

it's called as `labelling network` or `target specific network` in the paper.
"""

from torch import nn
import torchvision.models as models


class ClassifierA(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, nbClasses, dropout_keep=0.5, use_BN=True, inputSize = 768):
        """Init classifier."""
        super(ClassifierA, self).__init__()

        self.dropout_keep = dropout_keep
        self.use_BN = use_BN
        self.restored = False

        if use_BN:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(inputSize, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(100, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, nbClasses),
                nn.Softmax()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(inputSize, 100),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, nbClasses),
                nn.Softmax()
            )

    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out


class ClassifierB(nn.Module):
    """Feature classifier class for MNIST -> SVHN experiment in ATDA."""

    def __init__(self, nbClasses):
        """Init classifier."""
        super(ClassifierB, self).__init__()

        self.restored = False

        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, nbClasses),
            nn.Softmax()
        )

    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out


class ClassifierC(nn.Module):
    """Feature classifier class for SVHN -> MNIST or SYN Digits -> SVHN."""

    def __init__(self, nbClasses, dropout_keep=0.5, use_BN=True):
        """Init classifier."""
        super(ClassifierA, self).__init__()

        self.dropout_keep = dropout_keep
        self.use_BN = use_BN
        self.restored = False

        if self.use_BN:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(3072, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(2048, nbClasses),
                nn.BatchNorm1d(nbClasses),
                nn.Softmax()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_keep),
                nn.Linear(3072, 2048),
                nn.ReLU(),
                nn.Dropout(self.dropout_keep),
                nn.Linear(2048, nbClasses),
                nn.Softmax()
            )

    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out


class ClassifierVGG(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, nbClasses, dropout_keep=0.5, use_BN=True, inputSize = 768):
        """Init classifier."""
        super(ClassifierVGG, self).__init__()

        cl = models.__dict__['vgg16'](pretrained=True).classifier
        lastLayerIdx = list(cl._modules.keys())[-1]
        lastLayer = cl._modules[lastLayerIdx]
        inSize = lastLayer.in_features
        cl._modules[lastLayerIdx] = nn.Linear(inSize, nbClasses, bias=True)
        self.classifier = cl

        self.dropout_keep = dropout_keep
        self.use_BN = use_BN
        self.restored = False

    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out
