import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

# Set device
base_dir = './' # change to your hf model directory
device = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = ['baseline', 'gate_elementwise', 'gate_headwise']


try:
    import ao_fast_decode
except ImportError:
    ao_fast_decode = None

_GENERATE_MODES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': None}


def _generate_dtype():
    name = os.environ.get('GATED_ATTENTION_OPT_2', 'fp16').lower()
    if name not in _GENERATE_MODES:
        raise SystemExit('GATED_ATTENTION_OPT_2 must be one of: %s'
                         % ', '.join(sorted(_GENERATE_MODES)))
    return _GENERATE_MODES[name]


def attention_map_demo():
    # Loop through different model variants
    for name in VARIANTS:

        # Load model and tokenizer
        model_name_or_path = f"{base_dir}/1B_{name}"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)

        # Input text
        prompt = "Sparse gating mechanism mitigates attention sink."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Forward pass with output_attentions=True to retrieve attention scores
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True  # Retrieve attention scores
            )

        # Extract attention scores
        attentions = outputs.attentions  # tuple of tensors: (layer) -> (batch, head, seq_len, seq_len)

        # Function to average attention scores across all heads for each layer
        def average_heads(attentions):
            averaged = []
            for layer_attn in attentions:
                # layer_attn: (batch, head, seq_len, seq_len)
                avg_attn = layer_attn.mean(dim=1).cpu().numpy()  # (batch, seq_len, seq_len)
                averaged.append(avg_attn[0])  # Take the first sample
            return averaged

        averaged_attentions = average_heads(attentions)

        # Get tokens for axis labels
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Visualize attention maps of selected layers
        layers_to_visualize = [0, 6, 20, 27]  # Python indices start at 0, corresponds to 1st, 7th, 21st, 28th layers
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, layer_idx in enumerate(layers_to_visualize):
            attn_map = averaged_attentions[layer_idx]

            # Plot attention map
            ax = axes[idx]
            im = ax.imshow(attn_map, cmap="viridis")

            # Add colorbar
            fig.colorbar(im, ax=ax)

            # Set title
            ax.set_title(f"Layer {layer_idx + 1}")

            # Set ticks and labels
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)

            # Hide tick marks
            ax.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()
        plt.savefig(f"{name}_selected_layer_attention_maps.png")
        plt.show()


def read_prompts(path):
    with open(path, encoding='utf-8') as handle:
        return [line.rstrip('\n') for line in handle if line.strip()]


def generate_demo(args):
    model_name_or_path = f"{base_dir}/1B_{args.variant}"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    load_kwargs = {'trust_remote_code': True}
    dtype = _generate_dtype()
    if dtype is not None:
        load_kwargs['torch_dtype'] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map={'': device}, **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs).to(device)
    model.eval()
    fast = ao_fast_decode.install(model) if ao_fast_decode else None
    cache_impl = ao_fast_decode.cache_implementation() if ao_fast_decode else None

    prompts = read_prompts(args.prompts)
    size = max(1, args.batch_size)
    batches = [prompts[i:i + size] for i in range(0, len(prompts), size)]
    out_handle = open(args.out, 'w', encoding='utf-8')

    for unit_index, batch_prompts in enumerate(batches):
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                cache_implementation=cache_impl,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = tokenizer.batch_decode(
            sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        for completion in completions:
            out_handle.write(completion.replace('\n', ' ') + '\n')
        out_handle.flush()
        print(f"[gen] unit {unit_index + 1}/{len(batches)}", flush=True)

    out_handle.close()
    if fast is not None:
        fast.remove()
    print(f"[gen] wrote {len(prompts)} completions to {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description="Gated-attention demo: attention maps (default) or batch text generation.")
    parser.add_argument('--generate', action='store_true',
                        help='generate text for a file of prompts instead of plotting attention maps')
    parser.add_argument('--variant', default='gate_headwise', choices=VARIANTS,
                        help='which checkpoint to generate with (--generate only)')
    parser.add_argument('--prompts', default=None,
                        help='text file with one prompt per line (--generate only)')
    parser.add_argument('--max-new-tokens', type=int, default=96,
                        help='tokens to generate per prompt (--generate only)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='prompts per generate call (--generate only)')
    parser.add_argument('--out', default='generated_outputs.txt',
                        help='where completions are written (--generate only)')
    args = parser.parse_args()

    if args.generate:
        if not args.prompts:
            parser.error('--generate needs --prompts FILE (one prompt per line)')
        generate_demo(args)
    else:
        attention_map_demo()


if __name__ == '__main__':
    main()
