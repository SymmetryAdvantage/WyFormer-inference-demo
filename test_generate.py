import argparse
import json
import torch
import pytest
from pathlib import Path
from main import generate, GenerationMode


def _default_args(output_file, **overrides):
    args = argparse.Namespace(
        output_file=output_file,
        hf_model="kazeevn/WyFormer-MP20",
        device=torch.device("cpu"),
        initial_n_samples=10,
        firm_n_samples=None,
        generate_mode=GenerationMode.WyckoffJSONs,
        csx=False,
        required_elements=None,
        allowed_elements="all",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_wyckoff_jsons_format(tmp_path):
    output_file = tmp_path / "output.json"
    generate(_default_args(output_file))

    assert output_file.exists()
    with open(output_file, encoding="ascii") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) > 0
    for entry in data:
        assert isinstance(entry["group"], int)
        assert 1 <= entry["group"] <= 230
        assert isinstance(entry["species"], list)
        assert all(isinstance(s, str) for s in entry["species"])
        assert isinstance(entry["numIons"], list)
        assert all(isinstance(n, int) for n in entry["numIons"])
        assert isinstance(entry["sites"], list)
        assert all(isinstance(site, list) for site in entry["sites"])
        n_species = len(entry["species"])
        assert len(entry["numIons"]) == n_species
        assert len(entry["sites"]) == n_species


def test_wyckoff_jsons_firm_n_samples(tmp_path):
    output_file = tmp_path / "output.json"
    generate(_default_args(output_file, initial_n_samples=50, firm_n_samples=2))

    with open(output_file, encoding="ascii") as f:
        data = json.load(f)

    assert len(data) == 2


def test_wyckoff_tensors_format(tmp_path):
    output_file = tmp_path / "output.pt"
    generate(_default_args(output_file, generate_mode=GenerationMode.WyckoffTensors))

    assert output_file.exists()
    data = torch.load(output_file, weights_only=True)

    assert "start_tokens" in data
    assert "generated_tensors" in data
    assert "cascade_order" in data
    assert isinstance(data["cascade_order"], list)
    assert data["start_tokens"].shape[0] == data["generated_tensors"].shape[0]


def test_csx_requires_elements(tmp_path):
    output_file = tmp_path / "output.json"
    with pytest.raises(ValueError, match="--required-elements"):
        generate(_default_args(output_file, csx=True, required_elements=None))


def _load_json(path):
    with open(path, encoding="ascii") as f:
        return json.load(f)


def test_csx_required_elements_always_present(tmp_path):
    """Every generated structure must contain all required elements."""
    output_file = tmp_path / "output.json"
    generate(_default_args(
        output_file,
        csx=True,
        required_elements="Li-O",
        initial_n_samples=20,
    ))

    data = _load_json(output_file)
    assert len(data) > 0
    for entry in data:
        species = set(entry["species"])
        assert "Li" in species, f"Li missing from {species}"
        assert "O" in species, f"O missing from {species}"


def test_csx_fix_allowed_restricts_to_required(tmp_path):
    """With allowed_elements='fix', species must be exactly the required elements."""
    output_file = tmp_path / "output.json"
    generate(_default_args(
        output_file,
        csx=True,
        required_elements="Li-O",
        allowed_elements="fix",
        initial_n_samples=20,
    ))

    data = _load_json(output_file)
    assert len(data) > 0
    allowed = {"Li", "O"}
    for entry in data:
        species = set(entry["species"])
        assert "Li" in species, f"Li missing from {species}"
        assert "O" in species, f"O missing from {species}"
        assert species <= allowed, f"Unexpected elements {species - allowed}"


def test_csx_custom_allowed_elements(tmp_path):
    """Species must be within the allowed set, and required elements always appear."""
    output_file = tmp_path / "output.json"
    generate(_default_args(
        output_file,
        csx=True,
        required_elements="Li",
        allowed_elements="Li-O-Na",
        initial_n_samples=20,
    ))

    data = _load_json(output_file)
    assert len(data) > 0
    allowed = {"Li", "O", "Na"}
    for entry in data:
        species = set(entry["species"])
        assert "Li" in species, f"Li missing from {species}"
        assert species <= allowed, f"Unexpected elements {species - allowed}"


def test_csx_firm_n_samples(tmp_path):
    """firm_n_samples subsampling works in CSX mode."""
    output_file = tmp_path / "output.json"
    generate(_default_args(
        output_file,
        csx=True,
        required_elements="Li-O",
        initial_n_samples=50,
        firm_n_samples=2,
    ))

    data = _load_json(output_file)
    assert len(data) == 2
    for entry in data:
        assert "Li" in entry["species"]
        assert "O" in entry["species"]


def test_csx_wyckoff_tensors(tmp_path):
    """CSX mode in WyckoffTensors produces a valid tensor file."""
    output_file = tmp_path / "output.pt"
    generate(_default_args(
        output_file,
        csx=True,
        required_elements="Li-O",
        generate_mode=GenerationMode.WyckoffTensors,
        initial_n_samples=20,
    ))

    assert output_file.exists()
    data = torch.load(output_file, weights_only=True)
    assert "start_tokens" in data
    assert "generated_tensors" in data
    assert "cascade_order" in data
    assert data["start_tokens"].shape[0] == data["generated_tensors"].shape[0]


def _read_cif_structures(path):
    from pymatgen.io.cif import CifParser
    return CifParser(path).parse_structures(primitive=False)


def test_unrelaxed_structures_produces_cif(tmp_path):
    output_file = tmp_path / "output.cif"
    generate(_default_args(
        output_file,
        generate_mode=GenerationMode.UnrelaxedStructures,
        initial_n_samples=50,
    ))

    structures = _read_cif_structures(output_file)
    assert len(structures) > 0
    for s in structures:
        assert len(s) > 0


def test_unrelaxed_structures_firm_n_samples(tmp_path):
    """firm_n_samples limits structures passed to pyxtal, not necessarily CIF entries."""
    output_file = tmp_path / "output.cif"
    generate(_default_args(
        output_file,
        generate_mode=GenerationMode.UnrelaxedStructures,
        initial_n_samples=100,
        firm_n_samples=5,
    ))

    structures = _read_cif_structures(output_file)
    assert 1 <= len(structures) <= 5


def test_unrelaxed_structures_composition_preserved(tmp_path):
    """Elements in generated CIF structures match the required CSX element set."""
    output_file = tmp_path / "output.cif"
    generate(_default_args(
        output_file,
        generate_mode=GenerationMode.UnrelaxedStructures,
        csx=True,
        required_elements="Li-O",
        allowed_elements="fix",
        initial_n_samples=50,
    ))

    allowed = {"Li", "O"}
    structures = _read_cif_structures(output_file)
    assert len(structures) > 0
    for s in structures:
        elements = {el.symbol for el in s.composition.elements}
        assert "Li" in elements, f"Li missing from {elements}"
        assert "O" in elements, f"O missing from {elements}"
        assert elements <= allowed, f"Unexpected elements {elements - allowed}"
