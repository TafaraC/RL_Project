# Competition notebooks

Each notebook runs one frozen agent through the official ARC-AGI-3 Agent interface. The trained weights for all nine are in the `model_dataset` folder.

## Notebooks

| Notebook | Agent | Training seed | Click grid | Action choice |
|---|---|---|---|---|
| **`dqn_baseline.ipynb`** | **DQN baseline (nominated entry)** | 1 | 8×8 | epsilon 0.3 |
| `dqn_novelty.ipynb` | DQN + novelty bonus | 0 | 8×8 | epsilon 0.3 |
| `dqn_noop.ipynb` | DQN + no-change memory | 1 | 8×8 | epsilon 0.3 |
| `dqn_combined.ipynb` | DQN + novelty + no-change memory | 0 | 8×8 | epsilon 0.3 |
| **`ppo_novelty.ipynb`** | **PPO + novelty bonus (PPO alternative)** | 2 | 8×8 | sampled |
| `ppo_baseline.ipynb` | PPO baseline | 2 | 8×8 | sampled |
| `ppo_grid_control.ipynb` | PPO + novelty, finer grid | 1 | 16×16 | sampled |
| `ppo_memory_novelty.ipynb` | PPO + novelty + action memory | 0 | 16×16 | sampled |
| `ppo_memory_only.ipynb` | PPO + action memory | 2 | 16×16 | sampled |

All agents were trained on ft09 and play up to 1,000 actions per game. For each condition, the training seed with the most ft09 development successes was chosen, with ties going to the lowest seed. Held-out results played no part in the choice. Weights stay fixed during play, and memories reset between games.

## Running a notebook on Kaggle

1. **Upload the weights.** Create a Kaggle dataset and upload the contents of the `model_dataset` folder, keeping its subfolders as they are.
2. **Open the notebook.** Import the notebook into the ARC Prize 2026 – ARC-AGI-3 competition.
3. **Attach both datasets:** the official competition dataset and the uploaded model dataset.
4. **Set the notebook options:**
   - Accelerator: GPU T4.
   - Internet: off.
   - Use Kaggle's own Python environment; do not install `source/requirements.txt` there.
5. **Run it.** Choose Save & Run All. The notebook prints its Python, PyTorch and GPU versions, loads the weights and takes one test action.
6. **Submit.** Choose Submit to Competition and select `submission.parquet`. The score comes from the competition's rerun. The file written during the ordinary run is only a placeholder.

The inference code (`source/kaggle/my_agent.py` and `runtime.py`) is embedded in each notebook, so no other download is needed. Running the code needs only `arc-agi` and `python-dotenv` from the competition's offline packages, plus Kaggle's preinstalled PyTorch, NumPy and pandas.

## Runtime checks

Local training used Windows PyTorch 2.14.1+cu130. Kaggle uses its own Linux/Python environment and installed PyTorch; do not upload or install the Windows wheel. The notebooks use float32 NumPy weights and ordinary PyTorch layers. Kaggle must provide NumPy, PyTorch, pandas and parquet support. Missing CUDA is a fatal error.

All nine notebooks passed local CUDA inference and official Agent-loop checks on the development games. The nominated notebook also wrote and read its commit-mode parquet locally. Competition offline installation, the Kaggle Linux image and the live gateway have not been tested. No official score or submission identifier is available.
