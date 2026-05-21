#!/usr/bin/env python3
"""
CLI pour le pipeline d'analyse audio Arborisis.

Usage:
    python cli.py --input audio.wav --output ./out --config '{"n_fft":2048}'
"""

import argparse
import json
import os
import sys

from pipeline import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyse audio Arborisis")
    parser.add_argument("--input", "-i", required=True, help="Chemin du fichier audio")
    parser.add_argument("--output", "-o", default="./output", help="Répertoire de sortie")
    parser.add_argument("--config", "-c", default="{}", help="Config JSON (string)")
    parser.add_argument("--pretty", "-p", action="store_true", help="JSON formaté")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(json.dumps({"success": False, "error": f"Fichier introuvable: {args.input}"}))
        sys.exit(1)

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Config JSON invalide: {e}"}))
        sys.exit(1)

    try:
        result = run_analysis(args.input, args.output, config)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
