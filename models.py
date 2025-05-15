
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) for ECG classification.
    It consists of two convolutional layers followed by max pooling and two fully connected layers.
    The sigmoid activation is used for the output layer, assuming a binary classification task.
    """
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
    
    @staticmethod
    def name():
        return "CNN"
    
class ECGLSTM(nn.Module):
    """
    A Long Short-Term Memory (LSTM) network for ECG classification.
    It consists of an LSTM layer followed by two fully connected layers and a sigmoid output.
    It processes the input sequence to capture temporal dependencies.
    """
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
    
    @staticmethod
    def name():
        return "LSTM"
    
class ECGRNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) for ECG classification.
    It uses a simple RNN layer followed by two fully connected layers and a sigmoid output.
    It processes the input sequence to learn temporal patterns.
    """
    def __init__(self, input_length, output=2):
        super(ECGRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear(input_length * 128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 1, L) -> (batch, L, 1)

        out, hn = self.rnn(x)
        out = out.contiguous().view(out.size(0), -1)

        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x
    
    @staticmethod
    def name():
        return "RNN"

class ECGGRU(nn.Module):
    """
    A Gated Recurrent Unit (GRU) network for ECG classification.
    It employs a GRU layer followed by two fully connected layers and a sigmoid output.
    GRUs are designed to capture temporal dependencies in sequential data.
    """
    def __init__(self, input_length, output=2):
        super(ECGGRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear(input_length * 128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 1, L) -> (batch, L, 1)

        out, hn = self.gru(x)
        out = out.contiguous().view(out.size(0), -1)

        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x
    
    @staticmethod
    def name():
        return "GRU"
    
class ECGHYBRID_LSTM(nn.Module):
    """
    A hybrid model combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory 
    (LSTM) for ECG classification. It uses CNN layers to extract features from the input, 
    followed by LSTM layers to model temporal dependencies of these features.
    Finally, it uses fully connected layers and a sigmoid output for classification.
    """
    def __init__(self, input_length, output=2):
        super(ECGHYBRID_LSTM, self).__init__()
        
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
        x = self.sigmoid(self.fc2(x))
        return x

    @staticmethod
    def name():
        return "HYBRID_LSTM"
    

class ECGHYBRID_RNN(nn.Module):
    """
    A hybrid model combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks 
    (RNN) for ECG classification. It uses CNN layers for feature extraction followed by RNN layers 
    to process the temporal sequence of these features.
    The model ends with fully connected layers and a sigmoid output for classification.
    """
    def __init__(self, input_length, output=2):
        super(ECGHYBRID_RNN, self).__init__()

        self.input_length = input_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.rnn = nn.RNN(input_size=64, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear((input_length // 4) * 128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)  # (batch, C, L) -> (batch, L, C)
        
        out, hn = self.rnn(x)
        out = out.contiguous().view(out.size(0), -1)
        
        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x

    @staticmethod
    def name():
        return "HYBRID_RNN"


class ECGHYBRID_GRU(nn.Module):
    """
    A hybrid model combining Convolutional Neural Networks (CNNs) and Gated Recurrent Units (GRU) for ECG classification.
    CNN layers are used to extract spatial features, which are then processed by GRU layers to capture temporal dynamics.
    The model concludes with fully connected layers and a sigmoid output for classification.
    """
    def __init__(self, input_length, output=2):
        super(ECGHYBRID_GRU, self).__init__()

        self.input_length = input_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear((input_length // 4) * 128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # (batch, C, L) -> (batch, L, C)

        out, hn = self.gru(x)
        out = out.contiguous().view(out.size(0), -1)
        
        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x

    @staticmethod
    def name():
        return "HYBRID_GRU"



class ECGHYBRID_GRU_BN(nn.Module):
    """
    A hybrid model with Convolutional Neural Networks (CNNs), Batch Normalization (BN), 
    and Gated Recurrent Units (GRU) for ECG classification.
    Batch normalization is added after each convolutional layer to improve training stability and performance.
    The CNN layers extract features, the GRU layers model temporal dependencies, and the 
    final fully connected layers with sigmoid output perform classification.
    """
    def __init__(self, input_length, output=2):
        super(ECGHYBRID_GRU_BN, self).__init__()

        self.input_length = input_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear((input_length // 4) * 128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # (batch, C, L) -> (batch, L, C)

        out, hn = self.gru(x)
        out = out.contiguous().view(out.size(0), -1)
        
        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x

    @staticmethod
    def name():
        return "HYBRID_GRU_BN"
    

class ECGHYBRID_LSTM_BN(nn.Module):
    """
    A hybrid model with Convolutional Neural Networks (CNNs), Batch Normalization (BN), 
    and Long Short-Term Memory (LSTM) for ECG classification.
    Batch normalization is applied after each convolutional layer to stabilize and accelerate training.
    The CNN layers extract relevant features, the LSTM layers model the temporal evolution of these features, 
    and the final fully connected layers with sigmoid output perform the classification.
    """
    def __init__(self, input_length, output=2):
        super(ECGHYBRID_LSTM_BN, self).__init__()
        
        self.input_length = input_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear((input_length//4)*128, 64)
        self.fc2 = nn.Linear(64, output)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # x: (batch, 1, L) → (batch, L, 1)
        x = x.permute(0, 2, 1)
        
        # LSTM
        out, (hn, cn) = self.lstm(x)
        out = out.contiguous().view(out.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(out))
        x = self.sigmoid(self.fc2(x))
        return x

    @staticmethod
    def name():
        return "HYBRID_LSTM_BN"