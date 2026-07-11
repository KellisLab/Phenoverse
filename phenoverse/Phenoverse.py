#!/usr/bin/python

##############################################

# Manoj M Wagle (USydney; MIT CSAIL; Broad Institute)

##############################################


import argparse
import logging
import os
import shutil
import sys
import textwrap

Log_Format = "%(levelname)s - %(asctime)s - %(message)s \n"
logging.basicConfig(stream=sys.stderr, format=Log_Format, level=logging.INFO)
logger = logging.getLogger(__name__)

from .util import setup_seed


def build_welcome_message(max_width=78):
    doc_line = 'Documentation: https://kellislab.github.io/Phenoverse/'
    issues_line = 'Issues: https://github.com/KellisLab/Phenoverse/issues'
    min_width = max(len(doc_line), len(issues_line)) + 4

    terminal_width = shutil.get_terminal_size(fallback=(max_width, 24)).columns
    width = max(min(terminal_width, max_width), min_width)
    inner_width = width - 4

    paragraph = (
        'Thank you for using Phenoverse, an interpretable deep learning tool for '
        'learning sample representations and characterizing disease states in '
        'single-cell transcriptomics : )'
    )
    lines = textwrap.wrap(paragraph, width=inner_width)
    lines.append('')
    lines.append(doc_line)
    lines.append(issues_line)

    border = '+' + '-' * (width - 2) + '+'
    body = '\n'.join(f'| {line.ljust(inner_width)} |' for line in lines)
    return f'{border}\n{body}\n{border}'


class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    def format_help(self):
        help_text = super().format_help()
        return f'\n{build_welcome_message()}\n\n{help_text}'


def build_parser():
    parser = argparse.ArgumentParser("Phenoverse", formatter_class=CustomHelpFormatter)

    general = parser.add_argument_group("General")
    general.add_argument("--seed", type=int, default=42, help="seed")
    general.add_argument("--gpu", type=str, default="0", help="Please specify the GPU to use")

    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--train", help="Path to training dataset (.h5ad)")
    input_group.add_argument("--test", help="Path to test/query dataset (.h5ad)")
    input_group.add_argument("--phenotypelabel", default="disease", help="Phenotype label column in the AnnData object. Default: `disease`")
    input_group.add_argument("--samplecol", default="donor_id", help="Sample/donor ID column in the AnnData object. Default: `donor_id`")
    input_group.add_argument("--celltypecol", default="cell_type", help="Cell type column in the AnnData object. Default: `cell_type`")
    input_group.add_argument("--checkpoint", default="Phenoverse_best_model.pth", help="Model checkpoint path (output when training, input when testing). Default: `Phenoverse_best_model.pth`")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_dir", default=".", help="Output directory for test results. Default: current working directory")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--embedding_dim", type=int, default=128, help="Dimension of the cell embedding space")
    model_group.add_argument("--token_dim", type=int, default=128, help="Dimension of the token for each cell type")
    model_group.add_argument("--num_prototypes", type=int, default=4, help="Number of prototypes per cell type")
    model_group.add_argument("--encoder_hidden_dim", type=int, default=256, help="Hidden dimension of the cell encoder")
    model_group.add_argument("--encoder_blocks", type=int, default=2, help="Number of residual blocks in the cell encoder")
    model_group.add_argument("--n_heads", type=int, default=4, help="Number of attention heads in the Perceiver-based aggregator")
    model_group.add_argument("--n_latents", type=int, default=8, help="Number of latent tokens in the Perceiver-based aggregator")
    model_group.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    training = parser.add_argument_group("Training")
    training.add_argument("--batch_size", type=int, default=1, help="Batch size")
    training.add_argument("--test_size", type=float, default=0.2, help="Fraction of samples held out for validation during training")
    training.add_argument("--max_epochs", type=int, default=400, help="Maximum number of training epochs")
    training.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    training.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation steps")

    task = parser.add_argument_group("Task")
    task.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["train", "test"],
        help=(
            "`train` to train a Phenoverse model;\n"
            "`test` to run a trained Phenoverse model on new data\n"
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(build_welcome_message(), file=sys.stderr)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    setup_seed(args.seed)

    from .train import train_main
    from .test import test_main

    if args.setting == "train":
        if not args.train:
            parser.error("--train is required when --setting is 'train'")
        train_main(
            adata_path=args.train,
            label=args.phenotypelabel,
            sampleidcol=args.samplecol,
            celltypelabel=args.celltypecol,
            embedding_dim=args.embedding_dim,
            token_dim=args.token_dim,
            num_prototypes=args.num_prototypes,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_blocks=args.encoder_blocks,
            n_heads=args.n_heads,
            n_latents=args.n_latents,
            dropout=args.dropout,
            batch_size=args.batch_size,
            test_size=args.test_size,
            out_checkpoint=args.checkpoint,
            max_epochs=args.max_epochs,
            patience=args.patience,
            accum_steps=args.accum_steps,
        )
    elif args.setting == "test":
        if not args.test:
            parser.error("--test is required when --setting is 'test'")
        test_main(
            adata_path=args.test,
            checkpoint_path=args.checkpoint,
            sampleidcol=args.samplecol,
            celltypelabel=args.celltypecol,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
