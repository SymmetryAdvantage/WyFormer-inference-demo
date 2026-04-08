[![Pytest](https://github.com/SymmetryAdvantage/WyFormer-inference-demo/actions/workflows/pytest.yml/badge.svg)](https://github.com/SymmetryAdvantage/WyFormer-inference-demo/actions/workflows/pytest.yml)

# WyFormer Inference Demo

Minimal demo of using [WyckoffTransformer](https://github.com/SymmetryAdvantage/WyckoffTransformer) as a library

## Inference using WyckoffTransformer's built-in CLI
```bash
uv run wyformer-generate output.json --hf-model SymmetryAdvantage/WyFormer-Alex-MP20
```

## main.py features

- Generate novel crystal structures from a pre-trained model
- Three output formats: Wyckoff JSON, raw tensors, or unrelaxed CIF structures
- **Chemical System eXploration (CSX)** mode for element-constrained generation

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Handle your PyTorch installation separately. By default, uv will get whatever is on PyPI. For CPU-only environment, you can `cp uv.toml.cpu uv.toml`.
```bash
# Install project dependencies
uv sync
```

## Usage

```bash
uv run python main.py [output_file] [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `output_file` | `generated_structures.json` | Path to save results |
| `--hf-model` | `kazeevn/WyFormer-MP20` | Hugging Face model ID |
| `--device` | `cpu` | Inference device: `cpu` or `cuda` |
| `--initial-n-samples` | `1100` | Number of samples to generate before filtering |
| `--firm-n-samples` | *(all valid)* | Final number of samples after filtering |
| `--generate-mode` | `WyckoffJSONs` | Output format (see below) |
| `--csx` | — | Enable Chemical System eXploration mode |
| `--required-elements` | — | Elements that must appear in all structures (e.g. `Li-S`); required with `--csx` |
| `--allowed-elements` | `all` | Elements permitted in CSX mode: `all`, `fix` (required only), or a custom set like `Li-O-Na` |

### Output formats

- **`WyckoffJSONs`** — JSON with space group, species, ion counts, and Wyckoff sites (suitable for pyXtal)
- **`WyckoffTensors`** — PyTorch `.pt` file with raw tensors for downstream processing
- **`UnrelaxedStructures`** — CIF files for direct structural analysis

### Examples

```bash
# Generate 100 valid structures
uv run python main.py output.json --initial-n-samples 1000 --firm-n-samples 100

# Generate Li-S compounds only
uv run python main.py liS.json --csx --required-elements Li-S --allowed-elements fix

# Allow additional elements beyond the required ones
uv run python main.py liO.json --csx --required-elements Li-O --allowed-elements Li-O-Na-P

# Export raw tensors
uv run python main.py tensors.pt --generate-mode WyckoffTensors

# GPU inference
uv run python main.py output.json --device cuda
```

## Running Tests

```bash
uv run pytest
```
