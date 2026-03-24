import argparse
from pathlib import Path
from enum import Enum
import time
import json
import torch
from wyckoff_transformer.trainer import WyckoffTrainer
from wyckoff_transformer.generator import WyckoffGenerator

GenerationMode = Enum("GenerationMode", ["WyckoffTensors", "WyckoffJSONs", "UnrelaxedStructures"])


def generate(args):
    if args.csx and not args.required_elements:
        raise ValueError("--required-elements is required when --csx is enabled.")

    trainer = WyckoffTrainer.from_huggingface(
        args.hf_model,
        device=args.device)

    if args.generate_mode == GenerationMode.WyckoffJSONs:
        generation_start_time = time.time()
        if args.csx:
            print("--- Running in Chemical System eXploration (CSX) mode ---")
            generated_wp = trainer.generate_structures(
                n_structures=args.initial_n_samples,
                calibrate=False,
                required_element_set=args.required_elements,
                allowed_element_set=args.allowed_elements
            )
        else:
            print("--- Running in Default Generation mode ---")
            generated_wp = trainer.generate_structures(
                args.initial_n_samples,
                calibrate=False
            )
        generation_end_time = time.time()
        print(f"Generation in total took {generation_end_time - generation_start_time} seconds")

        if args.firm_n_samples is not None:
            if len(generated_wp) >= args.firm_n_samples:
                generated_wp = generated_wp[:args.firm_n_samples]
            else:
                raise ValueError("Not enough valid structures to subsample.")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "wt", encoding="ascii") as f:
            json.dump(generated_wp, f)
    elif args.generate_mode == GenerationMode.WyckoffTensors:
        generation_start_time = time.time()
        generator = WyckoffGenerator(
            trainer.model, trainer.cascade_order, trainer.cascade_is_target,
            trainer.token_engineers, trainer.masks_dict, trainer.max_sequence_length)
        if args.csx:
            print("--- Running in Chemical System eXploration (CSX) mode ---")
            start_tensor = trainer._sample_start_tokens_from_distribution(args.initial_n_samples)
            if 'elements' not in trainer.tokenisers:
                raise ValueError("Element vocabulary ('elements') not found in trainer.tokenisers.")
            generated_tensors = generator.generate_tensors(
                start=start_tensor,
                required_element_set=args.required_elements,
                allowed_element_set=args.allowed_elements,
                elements_vocab=trainer.tokenisers['elements']
            )
        else:
            print("--- Running in Default Generation mode ---")
            start_tensor = trainer._sample_start_tokens_from_distribution(args.initial_n_samples)
            generated_tensors = generator.generate_tensors(start_tensor, compute_validity=False)
        generation_end_time = time.time()
        print(f"Generation in total took {generation_end_time - generation_start_time} seconds")

        cascade_order = list(trainer.cascade_order)
        if trainer.cascade_order[-1] == "harmonic_site_symmetries":
            del generated_tensors[-1]
            cascade_order = cascade_order[:-1]
        generated_tensors = torch.stack(generated_tensors, dim=-1)

        if args.firm_n_samples is not None:
            if generated_tensors.size(0) >= args.firm_n_samples:
                start_tensor = start_tensor[:args.firm_n_samples]
                generated_tensors = generated_tensors[:args.firm_n_samples]
            else:
                raise ValueError("Not enough samples to subsample.")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "start_tokens": start_tensor.detach().cpu(),
            "generated_tensors": generated_tensors.detach().cpu(),
            "cascade_order": cascade_order,
        }, args.output_file)
    elif args.generate_mode == GenerationMode.UnrelaxedStructures:
        pass
        # TODO
    else:
        raise ValueError(f"Unsupported generation mode: {args.generate_mode}")


def main():
    parser = argparse.ArgumentParser(description="WyckoffTransformer Inference Demo")
    parser.add_argument("output_file", type=Path, help="Path to save the inference results", default="generated_structures.json", nargs='?')
    parser.add_argument("--hf-model", type=str, help="Hugging Face model identifier",
                        default="kazeevn/WyFormer-MP20")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="Device to run inference on (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--initial-n-samples", type=int, help="The number of samples to try"
        " before filtering out the invalid ones.", default=1100)
    parser.add_argument("--firm-n-samples", type=int, help="The number of samples after generation, "
        "subsampling the valid ones if necessary. If not specified, all valid samples will be returned.")
    parser.add_argument("--generate-mode", type=GenerationMode, choices=list(GenerationMode), default=GenerationMode.WyckoffJSONs,
                        help="The format of the generated structures. 'WyckoffTensors' will return the raw tensors, 'WyckoffJSONs' will return the Wyckoff positions in JSON format suitable for pyXtal, and 'UnrelaxedStructures' will return the unrelaxed structures")
    parser.add_argument("--csx", action="store_true",
                        help="Enable Chemical System eXploration (CSX).")
    parser.add_argument("--required-elements", "--r", type=str,
                        help="Required elements for CSX mode (e.g., 'Li-S'). Must be provided if --csx is used.")
    parser.add_argument("--allowed-elements", "--a", type=str, default="all",
                        help="Allowed elements for CSX mode: 'all', 'fix', or a custom set (e.g., 'Li-S-P-O').")
    args = parser.parse_args()
    try:
        generate(args)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
