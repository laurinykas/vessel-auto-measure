"""
main.py - Pagrindinis programos paleidimo failas

Paleidžia GUI arba CLI režimą.
"""

import sys
import argparse


def run_gui():
    """Paleidžia grafinę sąsają."""
    from PyQt5.QtWidgets import QApplication
    from gui import VesselsForm
    
    app = QApplication(sys.argv)
    window = VesselsForm()
    window.show()
    sys.exit(app.exec_())

def run_cli(args):
    """Paleidžia komandinės eilutės režimu (deleguoja į cli.py)."""
    import os
    from cli import process_single_image, process_directory

    if os.path.isdir(args.input):
        # Aplankas — apdoroti visus paveikslėlius
        process_directory(args.input, args.output)
    else:
        # Vienas failas
        if not os.path.exists(args.input):
            print(f"KLAIDA: Nerastas kelias: {args.input}")
            sys.exit(1)

        # Jei --output atrodo kaip aplankas (be plėtinio), įdėti ten paveikslėlį su tuo pačiu vardu
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        if args.output.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            output_base = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            output_base = os.path.join(args.output, base_name)

        success = process_single_image(args.input, output_base)
        sys.exit(0 if success else 1)


def main():
    """Pagrindinis įėjimo taškas."""
    parser = argparse.ArgumentParser(
        description="Vessel Auto Measure - Akies dugno kraujagyslių analizė",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pavyzdžiai:
  python main.py                          # Paleisti GUI
  python main.py --cli -i image.jpg       # Apdoroti paveikslėlį
  python main.py --cli -i input/ -o out/  # Apdoroti visą aplanką
        """
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Paleisti komandinės eilutės režimu (be GUI)'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Įvesties failas arba aplankas'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Išvesties failas arba aplankas (numatytasis: output)'
    )

    args = parser.parse_args()

    if args.cli:
        if not args.input:
            parser.error("--cli režimui reikalingas --input argumentas")
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()