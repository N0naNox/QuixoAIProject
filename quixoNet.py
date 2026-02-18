import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import json
import matplotlib.pyplot as plt

# Data preparation
def load_and_encode_data(file_path):
    """
    Loads data from JSON and converts it to One-Hot Encoded Tensors
    """
    print(f"Reading data from {file_path}...")
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    X_list = []
    Y_list = []

    for board_str, values in raw_data.items():
        

        # Encode: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
        board_vector = []
        for char in board_str:
            one_hot = [0.0, 0.0, 0.0]


            val = 0 if char == ' ' else 1 if char == 'X' else 2

            one_hot[val] = 1.0
            board_vector.extend(one_hot)

        X_list.append(board_vector)
        Y_list.append([values[0]])

    

    return torch.tensor(X_list), torch.tensor(Y_list)

# Model definition


class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(75, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = torch.relu(x)

        # Layer 2
        x = self.layer2(x)
        x = torch.relu(x)

        # Layer 3
        x = self.layer3(x)
        x = torch.relu(x)

        # Output layer
        x = self.output(x)
        x = torch.sigmoid(x)

        # We use Tanh at the end to squash the score between -1 and 1
        return torch.tanh(self.output(x))

# Training
def train(model, train_loader, test_loader, device, epochs=1000, learning_rate=0.01):
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_loss_history = []
    test_loss_history = []

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            # Forward pass
            y_pred = model(batch_X)
            # Calculate loss
            loss = loss_fn(y_pred, batch_Y)
            # Backward pass
            loss.backward()
            # Weight update
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        if epoch % 50 == 0:
            avg_test_loss = evaluate(model, test_loader, device)
            test_loss_history.append(avg_test_loss)
            print(f"Epoch {epoch} | Average Training Loss: {avg_loss:.5f} | Average Test Loss: {avg_test_loss:.5f}")

    return train_loss_history, test_loss_history

# Evaluation
def evaluate(model, loader, device):
    model.eval()  # Set model to evaluation mode
    loss_fn = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_Y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    model.train()  # Reset to training mode
    return avg_loss

# Main
if __name__ == "__main__":
    # # 0. Choose device
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    # print(f"Device selected: {device}")

    # # 1. Prepare Data
    # X, Y = load_and_encode_data('working_dir\\game_dict_100K_random.json')

    # # 2. Configure dataloader and partition into train and test sets
    # dataset = TensorDataset(X, Y)
    # train_size = int(len(dataset) * 0.7)
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # # 3. Instantiate Model
    # net = TicTacToeNet().to(device)

    # # 4. Execute Training
    # train_loss_history, test_loss_history = train(net, train_loader, test_loader, device)

    # # 5. Save the result
    # torch.save(net.state_dict(), "tictactoe_model.pth")
    # print("\nModel saved to tictactoe_model.pth")

    # # 6. Plot loss over epochs
    # plt.figure()
    # epoch_axis = list(range(len(train_loss_history)))
    # plt.plot(epoch_axis, train_loss_history, label = "Train Loss")

    # eval_axis = list(range(0, len(train_loss_history), 50))
    # plt.plot(eval_axis, test_loss_history, label = "Test Loss")

    # plt.title('Loss over epochs: train and test')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss (MSE)')
    # plt.legend()
    # plt.show()


    load_and_encode_data('states_greedy.json')