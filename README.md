# Fine-Tuning ESM-3 on Soluble High-Enrichment DARPin Sequences

This repository contains code and resources for fine-tuning [ESM-3](https://github.com/evolutionaryscale/esm) using a curated dataset of [DARPins (Designed Ankyrin Repeat Proteins)](https://en.wikipedia.org/wiki/DARPin) that exhibit both high solubility and high enrichment scores. Our objective is to adapt the ESM-3 protein language model for enhanced representation and generation of stable, high-affinity DARPin sequences. Fine-tuning is supported through full model training as well as parameter-efficient methods.

Low-Rank Adaptation (LoRA) and Weight-Decomposed LoRA (DoRA) have been implemented for efficient fine-tuning. These methods enable adaptation of ESM-3 with fewer trainable parameters, allowing the model to generalize effectively to the DARPin distribution while reducing computational overhead.

To apply LoRA or DoRA, use `esm3_lora.py`. For full model fine-tuning, use `ESM3_fullmodel_finetune.py`.

---

## 🔬 Project Objective

DARPins are engineered protein scaffolds with high specificity and stability, making them valuable for therapeutics and diagnostics. By fine-tuning ESM-3 with highly enriched, soluble DARPins, we aim to:

- Improve downstream property prediction (e.g., solubility, affinity)
- Enable generation of de novo DARPin-like sequences with better developability
- Benchmark model performance against held-out test sets and known DARPin libraries

---

## Running the DARPin–ESM-3 Pipeline

This section outlines the steps to prepare DARPin sequences and fine-tune the ESM-3 model using the provided scripts.

Step 0 – Installation
---------------------
Create a conda environment and activate it:

```bash
conda create -n esm3 python=3.9 -y
conda activate esm3
```

Then install esm and clone the repository:

    pip install esm
    git clone https://github.com/taugroup/esm3-darpins.git
    cd Scripts


Step 1 – Generate FASTA from Scored Sequences
---------------------------------------------
Run the FASTA generation script to select the top 100 sequences by score:

    python3 make_fasta.py

This script:
- Reads a cleaned CSV with columns `Sequence` and `Score`
- Sorts by Score
- Writes the top 100 sequences to `fasta/darpin.fasta` in FASTA format


Step 2 – Train the Model
------------------------
To fine-tune the model, run one of the scripts from `esm3-darpins/Scripts/`:

- **LoRA or DORA**:  
  Run `esm3_lora.py` to fine-tune with Low-Rank Adaptation (LoRA) or DORA.

    `python3 esm3_lora.py`

  Inside `esm3_lora.py`, the `wrap_with_lora(base_model)` function uses:

  ```python
  config = LoraConfig(
      r=LORA_RANK,
      lora_alpha=LORA_ALPHA,
      use_dora=True,  # use DORA (Weight-Decomposed Low-Rank Adaptation)
      target_modules=[
          ...
      ]
  )
Set `use_dora=True` for DORA or `False` for standard LoRA.

- **Full Model Fine-tuning**:  
    Run `ESM3_fullmodel_finetune.py` for standard fine-tuning of the full model.

        `python3 ESM3_fullmodel_finetune.py`

Step 3 – Run Inference
------------------------
    Run `esm3_model_inference.py` to perform inference on the fine-tuned model

        `python3 esm3_model_inference.py`

---

## 📊 Dataset

The training dataset includes curated DARPin sequences with:

* **Solubility** > threshold (experimentally validated or predicted)
* **Enrichment Score** > threshold (based on phage or yeast display rounds)

Supported formats: `FASTA`, `CSV` (with metadata)

---

## 📈 Evaluation Metrics

* Perplexity (on held-out sequences)
* AUC-ROC for solubility and enrichment classification tasks
* Sequence similarity to training data (diversity evaluation)
* Structural validity (AlphaFold, Rosetta)

---

## 📎 Citation

If you use this work, please cite:

```bibtex
@misc{esm3-darpin,
  title={Fine-Tuning ESM-3 on Soluble High-Enrichment DARPin Sequences},
  author={Your Name et al.},
  year={2025},
  howpublished={\url{https://github.com/taugroup/esm3-darpins}},
}
```

---

## 🤝 Acknowledgements

* Facebook AI Research for ESM-3
* Texas A\&M Digital Twin Lab
* Contributors and DARPin research community

---

## 📬 Contact

For questions, suggestions, or collaboration inquiries, please contact:
* 📧 \[[zahidhussain909@tamu.edu](mailto:zahidhussain909@tamu.edu)]
* 📧 \[[jtao@tamu.edu](mailto:jtao@tamu.edu)]
