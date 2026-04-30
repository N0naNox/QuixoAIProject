"""
Quixo AI Project – Main Script
===============================
This script:
  1. Generates 3 dictionaries (100K games each): Random, Greedy, Heuristic
  2. Runs performance tournaments (1000 games) for each agent type
  3. Compares unknown-board rates (100 games) for greedy with each dictionary
  4. Evaluates dictionary quality (score distribution)
"""

import json
import time
import os
from tournament import Tournament

DICT_SIZE = 100_000       # Number of games for dictionary generation
TOURNAMENT_SIZE = 1000    # Number of games for performance tournaments
UNKNOWN_TEST_SIZE = 100   # Number of games for unknown-rate analysis
MAX_MOVES_PER_GAME = 700
HEURISTIC_PROGRESS_INTERVAL = 1000
DICT_OPPONENT_MODE = 'RANDOM'
GREEDY_EPSILON = 0.1
HEURISTIC_EPSILON = 0.1


def load_dict(filename):
    """Load a dictionary from JSON file, or return empty dict if not found."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def generate_dictionary(filename, play_mode, num_games, epsilon=0.1,
                        opponent_play_mode=DICT_OPPONENT_MODE):
    """Generate a dictionary by running num_games and save to file."""
    print(f"\n{'='*60}")
    print(f"Generating {play_mode} dictionary ({num_games} games)...")
    print(f"{'='*60}")

    t = Tournament()
    start = time.time()
    progress_interval = HEURISTIC_PROGRESS_INTERVAL if play_mode == 'HEURISTIC' else None
    t.run(num_games, play_mode=play_mode, epsilon=epsilon,
          opponent_play_mode=opponent_play_mode,
          max_moves_per_game=MAX_MOVES_PER_GAME,
          progress_interval=progress_interval)
    elapsed = time.time() - start

    t.save_dict_to_file(filename)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Dictionary size: {len(t.states_dict)} boards")
    print(f"  Saved to: {filename}")
    return t.states_dict



def run_performance_tournament(states_dict, play_mode, label, epsilon=0.1):
    """Run a tournament and print results."""
    t = Tournament(states_dict=dict(states_dict))  # copy so we don't mutate
    unknown_rates = t.run(TOURNAMENT_SIZE, play_mode=play_mode, epsilon=epsilon)
    t.print_results(label)
    Tournament.print_unknown_stats(unknown_rates, label)
    return t


def run_unknown_rate_analysis(states_dict, label, epsilon=0.1):
    """Run 100 games and report unknown-rate stats."""
    t = Tournament(states_dict=dict(states_dict))
    unknown_rates = t.run(UNKNOWN_TEST_SIZE, play_mode='GREEDY', epsilon=epsilon)
    Tournament.print_unknown_stats(unknown_rates, label)
    return unknown_rates








def test_nn_vs_greedy():
    import json
    import os
    from tournament import Tournament
    from quixoNet import load_network
    import torch

    # Load heuristic dictionary
    if os.path.exists('states_heuristic.json'):
        with open('states_heuristic.json', 'r') as f:
            heuristic_dict = json.load(f)
        print(f"Loaded heuristic dictionary with {len(heuristic_dict)} states")
    else:
        print("Heuristic dictionary not found!")
        return

    # Load neural network model
    if os.path.exists('quixo_model.pth'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_network('quixo_model.pth', device)
        print(f"Loaded neural network model on {device}")
    else:
        print("Neural network model not found!")
        return

    # Run tournament 1: Greedy (X) with heuristic dict vs NN (O) with model
    print("\n" + "="*60)
    print("Tournament 1: Greedy (X) vs NN (O) - 1000 games")
    print("="*60)
    t1 = Tournament(states_dict=heuristic_dict)
    unknown_rates1 = t1.run(num_games=1000, play_mode='GREEDY', opponent_play_mode='NN', model=model, epsilon=0.1)
    t1.print_results("Greedy (X) vs NN (O)")
    Tournament.print_unknown_stats(unknown_rates1, "Greedy (X) vs NN (O)")
    
    # Run tournament 2: NN (X) with model vs Greedy (O) with heuristic dict
    print("\n" + "="*60)
    print("Tournament 2: NN (X) vs Greedy (O) - 1000 games")
    print("="*60)
    t2 = Tournament(states_dict=heuristic_dict)
    unknown_rates2 = t2.run(num_games=1000, play_mode='NN', opponent_play_mode='GREEDY', model=model, epsilon=0.1)
    t2.print_results("NN (X) vs Greedy (O)")
    Tournament.print_unknown_stats(unknown_rates2, "NN (X) vs Greedy (O)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Greedy (X) win rate when X has dict advantage: {t1.stats['VICTORY_X']/10:.1f}%")
    print(f"NN (X) win rate when X has dict advantage:     {t2.stats['VICTORY_X']/10:.1f}%")


if __name__ == "__main__":

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Generate dictionaries (skip if files already exist)
    # ──────────────────────────────────────────────────────────────────

    # Random dictionary
    random_dict = load_dict('states_random.json')
    if random_dict is None:
        random_dict = generate_dictionary('states_random.json', 'RANDOM', DICT_SIZE)
    else:
        print(f"\nLoaded existing random dictionary ({len(random_dict)} boards)")

    # Greedy dictionary (uses the random dictionary as base)
    greedy_dict = load_dict('states_greedy.json')
    if greedy_dict is None:
        greedy_dict = generate_dictionary('states_greedy.json', 'GREEDY', DICT_SIZE,
                                          epsilon=GREEDY_EPSILON)
    else:
        print(f"Loaded existing greedy dictionary ({len(greedy_dict)} boards)")

    # Heuristic dictionary
    heuristic_dict = load_dict('states_heuristic.json')
    if heuristic_dict is None:
        heuristic_dict = generate_dictionary('states_heuristic.json', 'HEURISTIC',
                                             DICT_SIZE, epsilon=HEURISTIC_EPSILON)
    else:
        print(f"Loaded existing heuristic dictionary ({len(heuristic_dict)} boards)")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Performance tournaments (1000 games each)
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "#" * 60)
    print("# PERFORMANCE TOURNAMENTS (1000 games each)")
    print("#" * 60)

    # Greedy agent with random dictionary
    run_performance_tournament(random_dict, 'GREEDY',
                               "Greedy + Random Dict")

    # Greedy agent with greedy dictionary
    run_performance_tournament(greedy_dict, 'GREEDY',
                               "Greedy + Greedy Dict")

    # Greedy agent with heuristic dictionary
    run_performance_tournament(heuristic_dict, 'GREEDY',
                               "Greedy + Heuristic Dict")

    # Heuristic agent with different dictionaries
    run_performance_tournament(greedy_dict, 'HEURISTIC',
                               "Heuristic + Greedy Dict", epsilon=HEURISTIC_EPSILON)
    
    run_performance_tournament(heuristic_dict, 'HEURISTIC',
                               "Heuristic + Heuristic Dict", epsilon=HEURISTIC_EPSILON)

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Unknown-rate analysis (100 games)
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "#" * 60)
    print("# UNKNOWN BOARD RATE ANALYSIS (100 games each)")
    print("#" * 60)

    run_unknown_rate_analysis(random_dict,
                              "Greedy with Random Dict")
    run_unknown_rate_analysis(greedy_dict,
                              "Greedy with Greedy Dict")
    run_unknown_rate_analysis(heuristic_dict,
                              "Greedy with Heuristic Dict")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Dictionary quality comparison
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "#" * 60)
    print("# DICTIONARY QUALITY COMPARISON")
    print("#" * 60)

    Tournament.print_dict_quality(random_dict, "Random")
    Tournament.print_dict_quality(greedy_dict, "Greedy")
    Tournament.print_dict_quality(heuristic_dict, "Heuristic")


