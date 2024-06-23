import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import random

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Define the Seq2Seq class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.6):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# Function to translate User's input
def translate_sentence(model, input_sentence, eng_tokenizer, fren_tokenizer):
    device = next(model.parameters()).device
    # Tokenize the input sentence
    input_sequence = [eng_tokenizer.word_index[word.lower()] for word in input_sentence.split()]
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(1).to(device)
    # Perform translation using beam search
    trsl_sequence = beam_search_decoder(model, input_tensor, fren_tokenizer.word_index, beam_width=3, max_len=50)
    # Convert tensor indices to list of integers
    trsl_sequence = trsl_sequence.tolist() if isinstance(trsl_sequence, torch.Tensor) else trsl_sequence
    trsl_words = [fren_tokenizer.index_word[idx] for idx in trsl_sequence]
    # Remove <sos> and <eos> tokens and join the words
    trsl_sentence = ' '.join(trsl_words[1:-1])
    return trsl_sentence

# Beam Search Decoder function
def beam_search_decoder(model, src, trg_vocab, beam_width=10, max_len=100):
    device = next(model.parameters()).device
    hidden, cell = model.encoder(src)
    start_token = trg_vocab['<sos>']
    end_token = trg_vocab['<eos>']
    beam = [(torch.tensor([start_token], device=device), 0, hidden, cell)]  # (sequence, score, hidden, cell)

    for _ in range(max_len):
        new_beam = []
        for seq, score, hidden, cell in beam:
            if seq[-1].item() == end_token:
                new_beam.append((seq, score, hidden, cell))
                continue
            
            hidden = hidden.squeeze(1)
            cell = cell.squeeze(1)
            
            output, hidden, cell = model.decoder(seq[-1], hidden, cell)
            
            hidden = hidden.unsqueeze(1)
            cell = cell.unsqueeze(1)
            
            if output.dim() == 1:
                output = output.unsqueeze(0)
                
            log_probs = F.log_softmax(output, dim=1)
            top_log_probs, top_indices = log_probs.topk(beam_width)

            for i in range(beam_width):
                new_seq = torch.cat([seq, top_indices[0, i].unsqueeze(0)])
                new_score = score + top_log_probs[0, i].item()
                new_beam.append((new_seq, new_score, hidden, cell))

        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beam[0][0]
    return best_seq

# Load and preprocess data
def load_data(eng_sentences, fren_sentences):
    eng_sentences = ['<sos> ' + sentence + ' <eos>' for sentence in eng_sentences]
    fren_sentences = ['<sos> ' + sentence + ' <eos>' for sentence in fren_sentences]
    
    eng_tokenizer = Tokenizer(filters='')
    eng_tokenizer.fit_on_texts(eng_sentences)
    eng_sequences = eng_tokenizer.texts_to_sequences(eng_sentences)
    
    fren_tokenizer = Tokenizer(filters='')
    fren_tokenizer.fit_on_texts(fren_sentences)
    fren_sequences = fren_tokenizer.texts_to_sequences(fren_sentences)
    
    max_eng_len = max([len(seq) for seq in eng_sequences])
    max_fren_len = max([len(seq) for seq in fren_sequences])
    
    eng_sequences = pad_sequences(eng_sequences, maxlen=max_eng_len, padding='post')
    fren_sequences = pad_sequences(fren_sequences, maxlen=max_fren_len, padding='post')
    
    return eng_sequences, fren_sequences, eng_tokenizer, fren_tokenizer

# Example sentences (replace with actual data)
eng_sentences = ["this is a test", "another test sentence", "Hello, how are you?"]
fren_sentences = ["c'est un test", "une autre phrase de test", "Bonjour, comment ça va?"]

# Preprocess the data
eng_sequences, fren_sequences, eng_tokenizer, fren_tokenizer = load_data(eng_sentences, fren_sentences)

# Define vocab sizes
input_dim = len(eng_tokenizer.word_index) + 1
output_dim = len(fren_tokenizer.word_index) + 1

# Define the model
emb_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.5

encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
decoder = Decoder(output_dim, emb_dim, hidden_dim, n_layers, dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(encoder, decoder, device).to(device)

# Training setup
def train_model(model, input_tensor, target_tensor, epochs=10, batch_size=32, learning_rate=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criteria = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(epochs):
        for i in range(0, len(input_tensor), batch_size):
            src = torch.tensor(input_tensor[i:i+batch_size], dtype=torch.long).transpose(0, 1).to(device)
            trg = torch.tensor(target_tensor[i:i+batch_size], dtype=torch.long).transpose(0, 1).to(device)
            
            optimizer.zero_grad()
            output = model(src, trg)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = criteria(output, trg)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Manually split data into training and validation sets
def train_val_split(data, val_ratio=0.2):
    data_len = len(data)
    val_len = int(data_len * val_ratio)
    indices = list(range(data_len))
    random.shuffle(indices)
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]
    return train_indices, val_indices

train_indices, val_indices = train_val_split(eng_sequences)

input_tensor_train = [eng_sequences[i] for i in train_indices]
target_tensor_train = [fren_sequences[i] for i in train_indices]

# Train the model
train_model(model, input_tensor_train, target_tensor_train)

# Save the model and tokenizers
torch.save(model.state_dict(), "seq2seq_model.pth")
with open('eng_tokenizer.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(eng_tokenizer.to_json(), ensure_ascii=False))
with open('fren_tokenizer.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(fren_tokenizer.to_json(), ensure_ascii=False))

# Translate function using beam search
def translate_with_beam_search(model, input_sentence):
    trsl_sentence = translate_sentence(model, input_sentence, eng_tokenizer, fren_tokenizer)
    return trsl_sentence

# User Interface
def main():
    print("Welcome to the English to French Translator!")
    print("Type 'exit' to quit from this task.")
    
    while True:
        user_input = input("Enter an English sentence for translation: ")
        
        if user_input.lower() == 'exit':
            print("Good bye!")
            break
        
        trsl_sentence = translate_with_beam_search(model, user_input)
        print("Translated Sentence:", trsl_sentence)

if __name__ == "__main__":
    main()