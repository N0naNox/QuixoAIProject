import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import json
import matplotlib.pyplot as plt

operation_mode = "TRAIN"  # Change to "TRAIN" to train the model, or "INFERENCE" to load and predict
EPOCHS = 300
EVAL_EVERY = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 4096
SEED = 42


def count_to_weight(count):
    """
    Convert a board occurrence count into a training weight.

    sqrt(count) is a practical compromise: common boards matter more,
    but extremely frequent boards do not completely dominate training.
    """
    return float(count) ** 0.5


def split_state_string(state_str):
    """Split a serialized state into player-to-move and 25-cell board string."""
    if len(state_str) >= 2 and state_str[1] == ':' and state_str[0] in {'X', 'O'}:
        return state_str[0], state_str[2:]
    return None, state_str


def encode_state(state_str):
    current_player, board_str = split_state_string(state_str)

    board_vector = []
    for char in board_str:
        val = 0 if char == ' ' else 1 if char == 'X' else 2
        one_hot = [0.0, 0.0, 0.0]
        one_hot[val] = 1.0
        board_vector.extend(one_hot)

    if current_player == 'X':
        player_vector = [1.0, 0.0]
    elif current_player == 'O':
        player_vector = [0.0, 1.0]
    else:
        player_vector = [0.5, 0.5]

    return board_vector + player_vector

# Data preparation
def load_and_encode_data(file_path):
    """
    Loads data from JSON and converts it to one-hot encoded tensors.
    Also returns a per-sample weight derived from the state's occurrence count.
    """
    print(f"Reading data from {file_path}...")
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    X_list = []
    Y_list = []
    W_list = []

    for state_str, values in raw_data.items():
        X_list.append(encode_state(state_str))
        Y_list.append([values[0]])
        W_list.append([count_to_weight(values[1])])

    return (
        torch.tensor(X_list, dtype=torch.float32),
        torch.tensor(Y_list, dtype=torch.float32),
        torch.tensor(W_list, dtype=torch.float32)
    )


def weighted_mse_loss(predictions, targets, weights):
    squared_error = (predictions - targets) ** 2
    weighted_error = squared_error * weights
    return weighted_error.sum() / weights.sum().clamp_min(1e-8)

# Model definition


class QuixoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 25 board cells x 3 one-hot values + 2 values for player-to-move.
        self.layer1 = nn.Linear(77, 128)
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
def train(model, train_loader, test_loader, device, epochs=EPOCHS, learning_rate=0.001, eval_every=EVAL_EVERY):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    train_loss_history = []
    test_eval_epochs = []
    test_loss_history = []

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_Y, batch_W in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            batch_W = batch_W.to(device)

            optimizer.zero_grad()
            # Forward pass
            y_pred = model(batch_X)
            # Calculate loss
            loss = weighted_mse_loss(y_pred, batch_Y, batch_W)
            # Backward pass
            loss.backward()
            # Weight update
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_loss)

        if epoch % eval_every == 0 or epoch == epochs - 1:
            avg_test_loss = evaluate(model, test_loader, device)
            test_loss_history.append(avg_test_loss)
            test_eval_epochs.append(epoch)
            scheduler.step(avg_test_loss)
            print(f"Epoch {epoch} | Average Training Loss: {avg_loss:.5f} | Average Test Loss: {avg_test_loss:.5f}")

       

    return train_loss_history, test_loss_history, test_eval_epochs

# Evaluation
def evaluate(model, loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch_X, batch_Y, batch_W in loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            batch_W = batch_W.to(device)
            predictions = model(batch_X)
            loss = weighted_mse_loss(predictions, batch_Y, batch_W)
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
    One-hot encodes a single Quixo state string.
    Preferred format: "X:<25 board chars>" or "O:<25 board chars>".
    """
    return encode_state(board_str)

# Perform inference to get score prediction
def predict_score(model, board_str, device):
    """
    Takes a trained model and a board string, and outputs the predicted score.
    """
    # 1. Encode the board using our helper function
    board_vector = encode_single_board(board_str)

    # 2. Convert to tensor and add a "batch" dimension (shape becomes [1, 77])
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
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Device selected: {device}")

        # 1. Prepare Data
        X, Y, W = load_and_encode_data('states_heuristic.json')

        # 2. Configure dataloader and partition into train and test sets
        dataset = TensorDataset(X, Y, W)

        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        generator = torch.Generator()
        generator.manual_seed(SEED)

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        # 2.5 Load as usual
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                      generator=generator)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Instantiate Model
        net = QuixoNet().to(device)

        # 4. Execute Training
        train_loss_history, test_loss_history, test_eval_epochs = train(
            net,
            train_loader,
            test_loader,
            device,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            eval_every=EVAL_EVERY
        )

        # 5. Save the result
        torch.save(net.state_dict(), "quixo_model.pth")
        print("\nModel saved to quixo_model.pth")

        # 6. Plot loss over epochs
        plt.figure()
        epoch_axis = list(range(len(train_loss_history)))
        plt.plot(epoch_axis, train_loss_history, label = "Train Loss")
        plt.plot(test_eval_epochs, test_loss_history, label = "Test Loss")

        plt.title('Loss over epochs: train and test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.show()

    elif operation_mode == "INFERENCE":
        model = load_network("quixo_model.pth", torch.device("cpu"))
        board = "X: O X          OX   OX X O"
        score = predict_score(model, board, torch.device("cpu"))
        print(f'{board}, {score}')