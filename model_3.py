#! /usr/bin/env python3
"""
Model for MNIST classification
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Model for MNIST classification
    """
    def __init__(self):
        """
        Initialize the model

        Layer	in	k	p	s	out	jin	jout r_in  r_out
        Conv_1	28	3	0	1	26	1	1	 1	   3
        Conv_2	26	3	0	1	24	1	1	 3	   5
        Conv_3	24	3	0	1	22	1	1	 5	   7
        Point_1	22	1	0	1	22	1	1	 7	   7
        MaxPool	22	2	0	2	11	1	2	 7	   8
        Conv_4	11	3	0	1	9	2	2	 8	   12
        Conv_5	9	3	0	1	7	2	2	 12	   16
        Conv_6	7	3	0	1	5	2	2	 16	   20
        Conv_7	5	3	1	1	5	2	2	 20	   24
        GAP	    5	5	0	1	1	2	2	 24	   32
        """
        super(Net, self).__init__()

        # Convolutional Block 1 Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.07)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.07)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.07)
        )

        # Transition Block Layers
        self.point1 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Convolutional Block 2 Layers
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.07)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.07)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.07)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.07)
        )

        # Transition Block 2 Layers
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass
        """
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Transition Block 1
        x = self.point1(x)
        x = self.pool1(x)

        # Convolutional Block 2
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # Transition Block 2
        x = self.gap(x)

        # Flatten
        x = x.view(-1, 10)

        # Softmax
        return F.log_softmax(x, dim=-1)
