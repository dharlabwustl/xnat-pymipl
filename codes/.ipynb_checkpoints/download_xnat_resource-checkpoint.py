#!/usr/bin/env python3
"""XNAT utility functions and a command-line entry point for Bash pipelines."""

import argparse
import os
from pathlib import Path

import xnat

import csv
from urllib.parse import urlparse
def download_scan_resource_folder(
    connection, csv_file, resource_name, extension, output_folder=None
):
    """Download matching files into the resource-named or supplied folder."""
    destination_dir = Path(
        resource_name if output_folder is None else output_folder
    )
    extension = "." + extension.lstrip(".").lower()

    return download_scan_resource_from_csv(
        connection,
        csv_file,
        resource_name,
        extension,
        destination_dir,
    )


def run_download_scan_resource_folder(args):
    with connect_xnat() as connection:
        download_scan_resource_folder(
            connection,
            args.csv_file,
            args.resource_dir,
            args.file_ext,
            args.output_folder,
        )


def register_download_scan_resource_folder(commands):
    download = commands.add_parser(
        "download-scan-resource-folder",
        help="Download scan files into a resource-named or custom folder",
    )
    download.add_argument("csv_file")
    download.add_argument("resource_dir")
    download.add_argument("file_ext", help="For example: .nii.gz or nii.gz")
    download.add_argument(
        "output_folder",
        nargs="?",
        default=None,
        help="Output folder; defaults to the resource directory name",
    )
    download.set_defaults(handler=run_download_scan_resource_folder)

def download_scan_resource_from_csv(
    connection, csv_file, resource_name, extension, output_dir
):
    # extension = "." + extension.lstrip(".").lower()
    output_dir = Path(output_dir)
    downloaded = []

    # Avoid downloading the same scan twice if it appears in multiple rows.
    session_scan_ids = dict.fromkeys(get_session_scan_ids(csv_file))

    for session_id, scan_id in session_scan_ids:
        experiment = connection.experiments.get(session_id)
        if experiment is None:
            print(f"Session not found: {session_id}")
            continue

        scan = experiment.scans.get(scan_id)
        if scan is None:
            print(f"Scan not found: {session_id}/{scan_id}")
            continue

        resource = scan.resources.get(resource_name)
        if resource is None:
            print(f"Resource not found: {session_id}/{scan_id}/{resource_name}")
            continue

        # Separate folders prevent filename collisions between scans.
        destination_dir = output_dir  # / session_id / scan_id

        for remote_file in resource.files.values():
            filename = Path(remote_file.name).name

            if filename.lower().endswith(extension):
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / filename

                remote_file.download(str(destination))
                downloaded.append(destination)
                print(f"Downloaded: {destination}")

    if not downloaded:
        print(f"No matching {extension} files found.")

    return downloaded

def get_session_scan_ids(csv_file):
    results = []

    with open(csv_file, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            uri = row["URI"].strip()
            parts = urlparse(uri).path.strip("/").split("/")

            session_id = parts[parts.index("experiments") + 1]
            scan_id = parts[parts.index("scans") + 1]

            results.append((session_id, scan_id))

    return results
def download_resource_files(connection, session_id, resource_name, extension, output_dir):
    experiment = connection.experiments.get(session_id)
    if experiment is None:
        raise ValueError(f"Session not found: {session_id}")

    resource = experiment.resources.get(resource_name)
    if resource is None:
        print(f"Resource not found: {resource_name}")
        return []

    # extension = "." + extension.lstrip(".").lower()
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
def run_download_scan_resource_from_csv(args):
    with connect_xnat() as connection:
        download_scan_resource_from_csv(
            connection,
            args.csv_file,
            args.resource_dir,
            args.file_ext,
            args.output_dir,
        )


def register_download_scan_resource_from_csv(commands):
    """Register downloading scan resource files using a CSV."""
    download = commands.add_parser(
        "download-scan-resource-from-csv",
        help="Download scan resource files using session and scan IDs from a CSV",
    )
    download.add_argument("csv_file")
    download.add_argument("resource_dir")
    download.add_argument("file_ext", help="For example: .nii.gz or nii.gz")
    download.add_argument("output_dir")
    download.set_defaults(handler=run_download_scan_resource_from_csv)
def get_project_name(connection, session_id):
    """Return the project ID associated with a session."""
    experiment = connection.experiments.get(session_id)

    if experiment is None:
        raise ValueError(f"Session not found: {session_id}")

    return experiment.project


def run_get_project_name(args):
    with connect_xnat() as connection:
        print(get_project_name(connection, args.session_id))


def register_get_project_name(commands):
    command = commands.add_parser(
        "get-project-name",
        help="Return the project ID associated with a session",
    )
    command.add_argument("session_id")
    command.set_defaults(handler=run_get_project_name)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    register_download_resource(commands)
    register_download_scan_resource_from_csv(commands)
    register_download_scan_resource_folder(commands)
    register_get_project_name(commands)
    # Register future commands here.

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
