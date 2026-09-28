<div align="center">
  <a href="https://autooptm.com"><img src=".autooptm/logo.png" width="96" alt="AutoOptm"></a>

  <h1>Gated Attention · optimized by <a href="https://autooptm.com">AutoOptm</a></h1>

  <p><b>3.80x faster end to end</b> on the command below, output verified against the stock program.</p>

  <p>
    <a href="https://autooptm.com"><img alt="speedup" src="https://img.shields.io/badge/end--to--end-3.80x-2ea44f"></a>
    <a href="https://github.com/qiuzh20/gated_attention/commit/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2"><img alt="base" src="https://img.shields.io/badge/upstream-f4c2a5f6ffd6-blue"></a>
    <img alt="card" src="https://img.shields.io/badge/measured%20on-RTX%204090-lightgrey">
  </p>
</div>

> This is a fork of [qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention) at commit
> [`f4c2a5f6ffd6`](https://github.com/qiuzh20/gated_attention/commit/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2) with the AutoOptm patch applied on top.
> **What is measured is a text-generation mode this fork adds to `demo.py`.** Upstream's `python demo.py`
> loads the three 1.7B checkpoints, runs one 11-token forward through each and saves attention-map
> figures; that command is unchanged here. The new, opt-in `python demo.py --generate --prompts FILE`
> runs the `1B_gate_headwise` checkpoint through `transformers`' `generate()` over a file of prompts.
> That mode is what was measured, and the "before" figure is the same mode without the optimisation.
> The optimisation was found, measured and verified automatically by [AutoOptm](https://autooptm.com);
> the patch is also kept verbatim at [`.autooptm/autooptm.patch`](.autooptm/autooptm.patch).

## The result

| | |
|---|---|
| **Command** | `python demo.py --generate --prompts prompts.txt` (32 real-text prompts of 140-610 tokens, 96 new tokens each, batch 1, `1B_gate_headwise`) |
| **Entry point** | `demo.py` (the `--generate` mode is added by this fork; `python demo.py` with no arguments is upstream's attention-map demo, unchanged) |
| **Unit measured** | one prompt: tokenize → greedy generation of 96 new tokens → decode → one line written to the output file |
| **Before (the same mode, without the optimisation)** | 2,776 ms per prompt (90.61 s for the timed loop; 101.75 s for the whole run including startup) |
| **After (this tree, all switches default ON)** | 735 ms per prompt (23.85 s for the timed loop; 34.07 s for the whole run including startup) |
| **Speedup** | **3.80x** end to end on RTX 4090 (timed loop; per prompt 3.78x, whole run including startup 2.99x), noise floor of the host 0.48% |
| **Output** | next-token probabilities within 0.0078 (max absolute difference) of the stock program's, relative L2 0.0028, PSNR 62 dB; verified on the pinned prompts and on held-out prompts the optimiser never saw (max difference 0.0075, worst 1% trimmed) |

### What changed

| File | Where | Gain |
|---|---|---|
| `modeling_qwen3.py`, `ao_fast_decode.py` (new) | `Qwen3RMSNorm.forward` | 1.245x |
| `ao_fast_decode.py` (new) | `install()` | 1.04x |
| `ao_fast_decode.py` (new) | `_FastDecode` | 1.60x |
| `demo.py` | `generate_demo()` | 1.94x |
| `demo.py` | `main()` / `generate_demo()` / `read_prompts()`: the new `--generate` mode | — (how the run is measured) |

Each gain is measured on top of the rows above it. The checkpoints on the Hugging Face Hub carry
their own copy of `modeling_qwen3.py` (loaded with `trust_remote_code`), which is why
`ao_fast_decode.py` applies the change to whichever copy is loaded. Everything above affects the
`--generate` mode only; every change is behind a switch that defaults on (see
[`.autooptm/autooptm.patch`](.autooptm/autooptm.patch)).

## Reproduce

```bash
git clone https://github.com/autooptm/gated_attention-ao.git
cd gated_attention-ao
pip install "transformers==4.46.3" accelerate matplotlib safetensors
# put the 1B_gate_headwise checkpoint from QwQZh/gated_attention on the Hub in ./1B_gate_headwise, then,
# with any text file of one prompt per line:
python demo.py --generate --prompts prompts.txt
```

The diff against upstream is one commit: `git log -1 -p` shows it, and
`git diff f4c2a5f6ffd6` is the same patch as `.autooptm/autooptm.patch`.

---

<div align="center"><sub>Optimized by <a href="https://autooptm.com">AutoOptm</a> — point it at a repository, get back a verified speedup and the patch.</sub></div>

---

# Gated Attention: Implementation and Visualization

This repository contains the implementation of **gated attention** mechanisms based on [Qwen3](https://github.com/QwenLM/Qwen3) model architecture, along with tools for visualizing attention maps. Our modifications are based on findings from recent research that demonstrate how applying **sparse, head-specific gating after Scaled Dot-Product Attention (SDPA)** can significantly improve performance, training stability, and long-context generalization. More details are in our paper [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708).


---

## 🆕 Updates

**2025-12-20** — We will release additional analyses and case studies on the effectiveness of Gated Attention, including visualizations and quantitative investigations into the model’s internal mechanisms to offer more intuitive insights into how and why gating works.

**2025-11-26** — Our paper, *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, has been awarded the **NeurIPS 2025 Best Paper Award**! 🎉  
This prestigious honor recognizes only 4 papers out of 5,290 accepted submissions, highlighting the foundational impact of our work on attention mechanism design.  
Official announcement: [NeurIPS 2025 Best Paper Awards](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)

**2025-09-18** — Our paper, *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, has been selected as an **Oral Presentation** at **NeurIPS 2025**, placing among the top 1.5% of submissions (77 out of 5,290 accepted papers). This recognition underscores the significance and novelty of our findings in rethinking attention gating for scalable, stable, and long-context LLMs.

**2025-09-10** — **Gated Attention** has been successfully integrated into the official **Qwen3-Next** architecture, as featured in Qwen’s latest research blog ([Read Here](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)) and deployed in the [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) model. This real-world adoption validates our core hypothesis: gating mechanisms significantly enhance **training stability** and **ultra-long-context performance** (up to 1M tokens).

---


## 📚 Introduction

Gating mechanisms have long been a cornerstone of neural network design, enabling dynamic control over information flow. In this work, we focus on integrating and evaluating these mechanisms within standard softmax attention layers of transformer models.

We introduce a **query-dependent sparse gate** after the SDPA output (`G1`), which modulates each attention head independently using a sigmoid function. This simple yet effective change:

- Introduces **non-linearity** into the low-rank transformation formed by value and output projections.
- Enables **input-dependent sparsity**, preventing the "attention sink" phenomenon where early tokens dominate attention distributions.
- Improves **training stability**, allowing larger learning rates.
- Enhances **long-context extrapolation**, showing significant gains on benchmarks like RULER.

---

## 📦 Models

We provide models follow Qwen3's architecture with different gating configurations:

- `baseline`: Standard attention without any gating.
- `gate_headwise`: Headwise gating applied after SDPA.
- `gate_elementwise`: Elementwise gating applied after SDPA.

These models are available at [huggingface repo](https://huggingface.co/QwQZh/gated_attention).

---


## 🧪 Demo Usage

A demo script is included to load a trained model and visualize attention maps with gating enabled.

### Requirements

```bash
pip install transformers matplotlib numpy torch
```

### Run Demo

```bash
python demo.py
```

This will produce a file named `{model_name}_selected_layer_attention_maps.png`, showing attention maps for four key layers.

#### Attention Maps Comparison

Below are the attention maps from **Layer 1**, **Layer 7**, **Layer 21**, and **Layer 28** of three different model variants: `baseline`, `gate_headwise`, and `gate_elementwise`. These visualizations help illustrate how gating mechanisms affect attention patterns, especially in relation to the "attention sink" phenomenon.

In the **baseline** model, we observe a strong "attention sink" effect — the **first token** consistently receives disproportionately high attention scores across multiple layers. This indicates that the model overly relies on the initial token, potentially limiting its ability to distribute attention meaningfully across other positions.

##### Baseline Model  

![baseline](baseline_selected_layer_attention_maps.png)

> **Observation**: Strong diagonal dominance with significant focus on the first token (attention sink). This pattern persists across multiple layers.

##### Gate Headwise Model  

![headwise](gate_headwise_selected_layer_attention_maps.png)

> **Observation**: Gating applied headwise reduces the attention sink effect. Attention becomes more distributed and context-dependent.

##### Gate Elementwise Model  

![elementwise](gate_elementwise_selected_layer_attention_maps.png)

> **Observation**: Elementwise gating further enhances sparsity and selectivity in attention patterns, leading to cleaner and more structured attention maps.

---

## 📁 Repository Structure

```sh
qwen3-gated/
├── figs/                    # used figs in the paper
├── modeling_qwen3.py        # Modified Qwen3 model with gated attention
├── configuration_qwen3.py   # Model configuration with gating flags
├── demo.py                  # Simple demo for loading model and extracting 
└── README.md                # You are here
```

---

## 🔧 Implementation Details

The core changes to implement gated attention are found in the `Qwen3Attention` class in the provided code.

### 🧠 Gating Variants

We support two main types of gating:

#### 1. **Headwise Gating**

Each attention head has its own gate scalar.

```python
self.headwise_attn_output_gate = True
```

These options can be configured in the model config under:

```json
{
  "headwise_attn_output_gate": true,
  "elementwise_attn_output_gate": false
}
```

#### 2. **Elementwise Gating**

Each element of the attention output is modulated independently.

```python
self.elementwise_attn_output_gate = True
```

These options can be configured in the model config under:

```json
{
  "headwise_attn_output_gate": false,
  "elementwise_attn_output_gate": true
}
```

---


## 📝 Citation

If you use this code or models in your research, please cite our paper:

```bibtex
@misc{qiu2025gatedattentionlargelanguage,
      title={Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free}, 
      author={Zihan Qiu and Zekun Wang and Bo Zheng and Zeyu Huang and Kaiyue Wen and Songlin Yang and Rui Men and Le Yu and Fei Huang and Suozhi Huang and Dayiheng Liu and Jingren Zhou and Junyang Lin},
      year={2025},
      eprint={2505.06708},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.06708}, 
}
```

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out at <qzh11628@gmail.com>.
