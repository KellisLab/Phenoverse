# Installation guide

[![PyPI version](https://img.shields.io/pypi/v/Phenoverse?color=teal&cacheSeconds=300)](https://pypi.org/project/Phenoverse/)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://kellislab.github.io/Phenoverse/)
![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/KellisLab/Phenoverse?tab=MIT-1-ov-file#readme)

### Pre-requisites

- Python >= 3.12

!!! note
    We recommend creating a separate environment such as **[Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#)** to avoid package conflicts.

### Installing Phenoverse

Install <code><span style="color: red;">Phenoverse</span></code> via pip:

```bash
pip install Phenoverse
```

### Verifying installation

To check the <code><span style="color: red;">Phenoverse</span></code> installation, please run:

```bash
phenoverse --help
```

You should see an output like this:

<div class="output-box">
<pre>

+----------------------------------------------------------------------------+
| Thank you for using Phenoverse, an interpretable deep learning tool for    |
| learning sample representations and characterizing disease states in       |
| single-cell transcriptomics : )                                            |
|                                                                            |
| Documentation: https://kellislab.github.io/Phenoverse/                     |
| Issues: https://github.com/KellisLab/Phenoverse/issues                     |
+----------------------------------------------------------------------------+

usage: Phenoverse [-h] [--seed SEED] [--gpu GPU] [--train TRAIN] [--test TEST]
                  [--phenotypelabel PHENOTYPELABEL] [--samplecol SAMPLECOL]
                  [--celltypecol CELLTYPECOL] [--checkpoint CHECKPOINT]
                  [--output_dir OUTPUT_DIR] [--embedding_dim EMBEDDING_DIM]
                  [--token_dim TOKEN_DIM] [--num_prototypes NUM_PROTOTYPES]
                  [--encoder_hidden_dim ENCODER_HIDDEN_DIM]
                  [--encoder_blocks ENCODER_BLOCKS] [--n_heads N_HEADS]
                  [--n_latents N_LATENTS] [--dropout DROPOUT]
                  [--batch_size BATCH_SIZE] [--test_size TEST_SIZE]
                  [--max_epochs MAX_EPOCHS] [--patience PATIENCE]
                  [--accum_steps ACCUM_STEPS] --setting {train,test}

options:
  -h, --help            show this help message and exit

General:
  --seed SEED           seed
  --gpu GPU             Please specify the GPU to use

Input:
  --train TRAIN         Path to training dataset (.h5ad)
  --test TEST           Path to test/query dataset (.h5ad)
  --phenotypelabel PHENOTYPELABEL
                        Phenotype label column in the AnnData object. Default: `disease`
  --samplecol SAMPLECOL
                        Sample/donor ID column in the AnnData object. Default: `donor_id`
  --celltypecol CELLTYPECOL
                        Cell type column in the AnnData object. Default: `cell_type`
  --checkpoint CHECKPOINT
                        Model checkpoint path (output when training, input when testing). Default: `Phenoverse_best_model.pth`

Output:
  --output_dir OUTPUT_DIR
                        Output directory for test results. Default: current working directory

Model:
  --embedding_dim EMBEDDING_DIM
                        Dimension of the cell embedding space
  --token_dim TOKEN_DIM
                        Dimension of the token for each cell type
  --num_prototypes NUM_PROTOTYPES
                        Number of prototypes per cell type
  --encoder_hidden_dim ENCODER_HIDDEN_DIM
                        Hidden dimension of the cell encoder
  --encoder_blocks ENCODER_BLOCKS
                        Number of residual blocks in the cell encoder
  --n_heads N_HEADS     Number of attention heads in the Perceiver-based aggregator
  --n_latents N_LATENTS
                        Number of latent tokens in the Perceiver-based aggregator
  --dropout DROPOUT     Dropout rate

Training:
  --batch_size BATCH_SIZE
                        Batch size
  --test_size TEST_SIZE
                        Fraction of samples held out for validation during training
  --max_epochs MAX_EPOCHS
                        Maximum number of training epochs
  --patience PATIENCE   Early stopping patience (epochs)
  --accum_steps ACCUM_STEPS
                        Gradient accumulation steps

Task:
  --setting {train,test}
                        `train` to train a Phenoverse model;
                        `test` to run a trained Phenoverse model on new data
</pre>
</div>

<br>

---
<p class="page-footer" style="text-align: left; font-size: 15px">
  Documentation by <a href="http://manojmw.github.io" target="_blank">Manoj M Wagle</a>
</p>
