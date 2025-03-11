import torch.nn as nn
import torch
class EmbeddingLSTM(nn.Module):
    def __init__(self, num_pc, num_delta_in, num_output_next, embed_dim, hidden_dim, output_dim, topPredNum, num_layers, dropout):
        # Layer structure is described in the paper
        super(EmbeddingLSTM, self).__init__()
        self.topPredNum = topPredNum
        # Define embedding layers
        self.pc_embed_layer = nn.Embedding(num_pc, embed_dim)
        self.delta_embed_layer = nn.Embedding(num_delta_in, embed_dim)
        # Define LSTM layer
        lstm_input_dim = embed_dim * 2
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        # Define output layer
        self.fc = nn.Linear(hidden_dim, num_delta_in)
        # Define dropout layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_c_state, target=None):
        pc, delta = x
        pc_embed = self.pc_embed_layer(pc)
        delta_embed = self.delta_embed_layer(delta)
        pc_delta_embed_out = torch.cat((pc_embed, delta_embed), dim = 2).to(torch.float32)
        lstm_out, (h_0, c_0) = self.lstm(pc_delta_embed_out, h_c_state)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # Only take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        outputs = self.fc(lstm_out)

        # outputs = self.dropout(outputs)

        outputs = self.sigmoid(outputs)

        state = (h_0, c_0)

        if target is not None:

            target = target.float()

            loss = nn.CrossEntropyLoss()(outputs, target) 

        else:
            loss = None
        
        _, preds = torch.topk(outputs, self.topPredNum, sorted=True)

        # Get top k predictions
        return preds, state, loss

    def predict(self, X, lstm_state):
        with torch.no_grad():
            preds, state, _ = self.forward(X, lstm_state)
            return preds, state