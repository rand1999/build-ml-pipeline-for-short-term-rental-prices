#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact.
    logger.info("Downloading artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logger.info("Loaded %d rows", len(df))

    # Drop price outliers (inclusive range)
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info("Kept %d rows with %s <= price <= %s", len(df), args.min_price, args.max_price)

    # Drop listings outside the NYC boundaries (fix for release 1.0.1)
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    logger.info("Dropping %d rows outside NYC boundaries", int((~idx).sum()))
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save the cleaned data and upload it to W&B
    output_path = "clean_sample.csv"
    df.to_csv(output_path, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_path)
    logger.info("Logging artifact %s", args.output_artifact)
    run.log_artifact(artifact)
    artifact.wait()

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the raw data artifact in W&B (for example sample.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned data artifact (for example clean_sample.csv)",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (for example clean_sample)",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum nightly price to keep (inclusive)",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum nightly price to keep (inclusive)",
        required=True,
    )

    args = parser.parse_args()
    go(args)
