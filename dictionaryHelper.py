import json

def print_lines(num_lines, dict):
    """Helper function to print the first N lines of a dictionary."""
    for i, (key, value) in enumerate(dict.items()):
        if i >= num_lines:
            break
        print(f"{key}: {value}")



if __name__ == "__main__":
    try:
        with open("states_random.json", "r") as f:
            loaded_dict = json.load(f)
        print("Dictionary loaded successfully for the agent.")
        print_lines(10, loaded_dict)
        # for key, value in list(loaded_dict.items())[:10]:
        #     print(f"{key}")

    except Exception as e:
        print("Failed to load dictionary. Starting with an empty dictionary.")
        loaded_dict = {}