from torch import nn

class Lstm_Regressor(nn.Module):
    
    def __init__(self,input_size,hidden_dim,out_dim,n_layers,dropout=.1):
        super(Lstm_Regressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_dim, 
                          num_layers=n_layers, 
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=True
                         )
        self.linear = nn.Linear(hidden_dim*2, out_dim)
        
    def forward(self, X_batch):
        output, hidden = self.lstm(X_batch,
                                  #torch.randn(n_layers, len(X_batch), hidden_dim)
                                 )
        output = self.linear(output[:,-1])
        
        return output