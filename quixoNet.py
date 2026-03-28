import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import json
import matplotlib.pyplot as plt

operation_mode = "INFERENCE"

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
        # Clean: "[001...]" -> "001..."
        #clean_board = board_str[1:-1]

        # Encode: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
        board_vector = []
        for char in board_str:
            val = 0 if char == ' ' else 1 if char == 'X' else 2
            one_hot = [0.0, 0.0, 0.0]
            one_hot[val] = 1.0
            board_vector.extend(one_hot)

        X_list.append(board_vector)
        Y_list.append([values[0]])

    return torch.tensor(X_list), torch.tensor(Y_list)

# Model definition


class QuixoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(75, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = torch.relu(x)

        # Layer 2
        x = self.layer2(x)
        x = torch.relu(x)

        # Output layer
        x = self.output(x)
        x = torch.sigmoid(x)

        return x

# Training
def train(model, train_loader, test_loader, device, epochs=2000, learning_rate=0.001):
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

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

# Load network weights from .pth file
def load_network(model_path, device):
    """
    Loads a saved QuixoNet model from a .pth file.
    """
    print(f"Loading model from {model_path}...")

    # 1. Instantiate a fresh model
    model = QuixoNet().to(device)

    # 2. Load the saved weights into the model
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 3. Set the model to evaluation mode
    model.eval()

    return model

# Encode board for input into network
def encode_single_board(board_str):
    """
    Cleans and one-hot encodes a single Tic-Tac-Toe board string.
    Example: "[102010201]" -> [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, ...]
    """
    #clean_board = board_str.strip("[]")

    board_vector = []
    for char in board_str:
        val = 0 if char == '0' else 1 if char == '1' else 2
        one_hot = [0.0, 0.0, 0.0]
        one_hot[val] = 1.0
        board_vector.extend(one_hot)

    return board_vector

# Perform inference to get score prediction
def predict_score(model, board_str, device):
    """
    Takes a trained model and a board string, and outputs the predicted score.
    """
    # 1. Encode the board using our helper function
    board_vector = encode_single_board(board_str)

    # 2. Convert to tensor and add a "batch" dimension (shape becomes [1, 27])
    x_tensor = torch.tensor([board_vector]).to(device)

    # 3. Make the prediction without calculating gradients
    with torch.no_grad():
        prediction = model(x_tensor)

    # 4. Extract the single float value from the resulting tensor
    return prediction.item()

# Main
if __name__ == "__main__":

    if operation_mode == "TRAIN":
        # 0. Choose device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Device selected: {device}")

        # 1. Prepare Data
        X, Y = load_and_encode_data('states_random.json')

        # 2. Configure dataloader and partition into train and test sets
        dataset = TensorDataset(X, Y)

        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        generator = torch.Generator()
        generator.manual_seed(42)

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        # 2.5 Load as usual
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # 3. Instantiate Model
        net = QuixoNet().to(device)

        # 4. Execute Training
        train_loss_history, test_loss_history = train(net, train_loader, test_loader, device)

        # 5. Save the result
        torch.save(net.state_dict(), "quixo_model.pth")
        print("\nModel saved to quixo_model.pth")

        # 6. Plot loss over epochs
        plt.figure()
        epoch_axis = list(range(len(train_loss_history)))
        plt.plot(epoch_axis, train_loss_history, label = "Train Loss")

        eval_axis = list(range(0, len(train_loss_history), 50))
        plt.plot(eval_axis, test_loss_history, label = "Test Loss")

        plt.title('Loss over epochs: train and test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.show()

    elif operation_mode == "INFERENCE":
        model = load_network("quixo_model.pth", torch.device("cpu"))
        board = "0000010000200000000000000"
        score = predict_score(model, board, torch.device("cpu"))
        print(f'{board}, {score}')