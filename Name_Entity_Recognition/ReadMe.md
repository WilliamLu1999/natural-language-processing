Hi, simply write "python3 somepath/hw4_Lu.py" in the terminal to run my code. Also, the glove.6B.100d.gz is decompressed already to glove.6B.100d in my code. So please be aware of that:)

embedding_dimension = 100
num_lstm_layer = 1
hidden_dimension = 256
dropout = 0.33
output_dimension = 128

bilstm1: lr=0.08, momentum=0.9,dampening=0.1, epoch number = 25, batch: 15 (all data)

bilstm2: lr=0.05, momentum=0.9, nesterov=True, epoch number = 25, batch: 15 (all data)

Thank you.