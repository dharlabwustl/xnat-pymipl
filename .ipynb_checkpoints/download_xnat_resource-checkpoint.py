#!/usr/bin/env python3
"""XNAT utility functions and a command-line entry point for Bash pipelines."""

import argparse
import os
from pathlib import Path

import xnat


def download_resource_files(connection, session_id, resource_name, extension, output_dir):
    experiment = connection.experiments.get(session_id)
    if experiment is None:
        raise ValueError(f"Session not found: {session_id}")

    resource = experiment.resources.get(resource_name)
    if resource is None:
        print(f"Resource not found: {resource_name}")
        return []

    extension = "." + extension.lstrip(".").lower()
    output_dir = Path(output_dir)
    downloaded = []

    for remote_file in resource.files.values():
        name = Path(remote_file.name).name
        if name.lower().endswith(extension):
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / name
            remote_file.download(str(destination))
            downloaded.append(destination)
            print(f"Downloaded: {destination}")

    if not downloaded:
        print(f"No {extension} files found in {resource_name}")
    return downloaded


def connect_xnat():
    """Connect using credentials exported by the calling shell."""
    return xnat.connect(
        os.environ["XNAT_HOST"],
        user=os.environ["XNAT_USER"],
        password=os.environ["XNAT_PASS"],
    )


def run_download_resource(args):
    with connect_xnat() as connection:
        download_resource_files(
            connection, args.session_id, args.resource_dir,
            args.file_ext, args.output_dir,
        )


def register_download_resource(commands):
    """Register the download-resource Bash command and its arguments."""
    download = commands.add_parser(
        "download-resource", help="Download files from a session-level resource"
    )
    download.add_argument("session_id")
    download.add_argument("resource_dir")
    download.add_argument("file_ext", help="For example: .nii.gz or nii.gz")
    download.add_argument("output_dir")
    download.set_defaults(handler=run_download_resource)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    register_download_resource(commands)
    # Register future commands here.

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
