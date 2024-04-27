import torch.nn as nn


class FlowModel(nn.Module):
             def __init__(self, input_size):
                 super(FlowModel, self).__init__()
                 self.fc1 = nn.Linear(input_size, 128)
                 self.dropout1 = nn.Dropout(0.5)
                 self.fc2 = nn.Linear(128, 64)
                 self.dropout2 = nn.Dropout(0.5)
                 self.fc3 = nn.Linear(64, 32)
                 self.dropout3 = nn.Dropout(0.5)
                 self.fc4 = nn.Linear(32, 1)

             def forward(self, x):
                 x = torch.sigmoid(self.fc1(x))
                 x = self.dropout1(x)
                 x = torch.sigmoid(self.fc2(x))
                 x = self.dropout2(x)
                 x = torch.sigmoid(self.fc3(x))
                 x = self.dropout3(x)
                 x = torch.sigmoid(self.fc4(x))
                 return x
