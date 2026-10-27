# WitsC Kaggle submission

Generated notebooks live in `submission/WitsC/kaggle/`. Each notebook embeds its complete inference source and losslessly exported model weights. No private dataset upload or GitHub download is needed to load a model.

## PyTorch and platform requirements

Training: Windows, Python 3.14.3, PyTorch 2.14.1+cu130, CUDA runtime 13.0, NVIDIA RTX 3050 Laptop GPU. These are training provenance, **not Kaggle installation requirements**.

Competition guidance specifies Python 3.12 on Linux. Use Kaggle's preinstalled CUDA-enabled PyTorch, NumPy, pandas and parquet support, and select a T4 GPU. Do not install the project's Windows wheel, `requirements.txt`, or `requirements-cuda.txt` inside Kaggle. Those pinned local versions are not portable competition dependencies.

Weights are exported to ordinary float32 NumPy arrays (`np.load(..., allow_pickle=False)`) and loaded into the same Conv2d/ReLU/Linear architecture. No PyTorch pickle, optimizer state, `torch.compile`, custom CUDA extension, Gymnasium or training module is needed for submission inference. Local validation compares exported-network outputs bit-for-bit with the original network.

The notebook prints actual Python/PyTorch/GPU versions, verifies asset hashes, imports the real official Agent class, loads all weights strictly and performs a legal CUDA action before writing any commit-mode placeholder. Missing CUDA or dependencies stop execution clearly. It uses competition-provided offline wheels only for `arc-agi` and `python-dotenv`.

Actual Kaggle image compatibility and gateway execution remain unverified until Save & Run All and competition rerun succeed. A local Windows GPU test is not equivalent to a Kaggle Linux test.

## Lecturer execution

1. Create a notebook in ARC Prize 2026 - ARC-AGI-3 using the account permitted to submit on WitsC's behalf.
2. Import the nominated `.ipynb`; attach the competition dataset, enable T4 GPU, disable internet.
3. Save & Run All. Inspect the printed preflight success and actual runtime versions.
4. Submit to Competition and select `submission.parquet`. The ordinary commit-mode file follows the official starter's placeholder schema and is **not a measured score**. During competition rerun, the official gateway records actions and creates the scored output.
5. Save the actual submission ID, timestamp, score and notebook URL for the team. Do not substitute the placeholder's numeric value for an official score.

Notebook code follows https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter and https://docs.arcprize.org/arc-prize-2026, retrieved 27 October 2026. We use the competition-provided framework rather than shipping development game sources.
