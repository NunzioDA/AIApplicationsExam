
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    def __init__(self, input_length, output = 2):
        super(ECGCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        conv_output_length = input_length // 4  
        

        self.fc1 = nn.Linear(64 * conv_output_length, 128) 
        self.fc2 = nn.Linear(128, output) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, 1, input_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
    def name(self):
        return "CNN"
    
class ECGLSTM(nn.Module):
    def __init__(self, input_length, output=2):
        super(ECGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)        

        self.fc1 = nn.Linear(input_length * 128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, 1, L) → (batch, L, 1)
        x = x.permute(0, 2, 1)
        
        # LSTM
        out, (hn, cn) = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x
    
    def name(self):
        return "LSTM"
    
class ECGHYBRID(nn.Module):
    def __init__(self, input_length, output=2):
        super(ECGHYBRID, self).__init__()
        
        self.input_length = input_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        

        self.fc1 = nn.Linear((input_length//4)*128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # x: (batch, 1, L) → (batch, L, 1)
        x = x.permute(0, 2, 1)
        
        # LSTM
        out, (hn, cn) = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x)) # logits
        return x

    def name(self):
        return "HYBRID"