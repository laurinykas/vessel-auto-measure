"""
cli.py - Komandinės eilutės sąsaja

"""
import argparse
import sys
import os
from pathlib import Path


def process_single_image(input_path: str, output_path: str):
    """
    Apdoroja vieną paveikslėlį.

    Args:
        input_path: Kelias iki paveikslėlio
        output_path: Išvesties bazinis kelias (be plėtinio) arba failas su plėtiniu.
                     Rezultatai bus išsaugoti kaip {base}_rez.png, {base}_rez.csv, {base}_rez2.csv
    """
    from processing import VesselProcessor

    print(f"Apdorojamas: {input_path}")

    def on_progress(percent, message):
        # Paprastas progreso rodiklis
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f"\r[{bar}] {percent:3d}% {message[:30]:30s}", end='', flush=True)

    processor = VesselProcessor()
    processor.set_progress_callback(on_progress)

    if not processor.load_image(input_path):
        print(f"\nKLAIDA: {processor.result.error_message}")
        return False

    result = processor.run_full_processing()
    print()  # Nauja eilutė po progreso

    if result.success:
        # Nuimti plėtinį, jei paduotas failo vardas — save_results pridės _rez.* pats
        base_path = os.path.splitext(output_path)[0]
        # Užtikrinti, kad išvesties aplankas egzistuoja
        parent = os.path.dirname(base_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        processor.save_results(base_path)
        print(f"Išsaugota: {base_path}_rez.png  (+ _rez.csv, _rez2.csv)")

        # Spausdinti trumpą santrauką
        if result.od_r > 0:
            print(f"  Optinis diskas: ({result.od_x}, {result.od_y}), r={result.od_r}")
        if result.avr:
            avr_total = result.avr.get('total', 0)
            if avr_total:
                print(f"  AVR: {avr_total:.4f}")
        return True
    else:
        print(f"KLAIDA: {result.error_message}")
        return False


def process_directory(input_dir: str, output_dir: str):
    """
    Apdoroja visus paveikslėlius aplanke.

    Args:
        input_dir: Įvesties aplankas
        output_dir: Išvesties aplankas
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Sukurti išvesties aplanką jei neegzistuoja
    output_path.mkdir(parents=True, exist_ok=True)

    # Palaikomi formatai
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ppm', '.bmp'}

    # Rasti visus paveikslėlius
    images = [f for f in input_path.iterdir()
              if f.is_file() and f.suffix.lower() in extensions]

    if not images:
        print(f"Nerasta paveikslėlių aplanke: {input_dir}")
        return

    print(f"Rasta {len(images)} paveikslėlių")
    print("-" * 60)

    success = 0
    failed = 0

    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] ", end='')

        # Išvesties bazinis kelias: {output_dir}/{stem}
        # save_results pats pridės _rez.png, _rez.csv, _rez2.csv
        out_base = output_path / img_path.stem

        if process_single_image(str(img_path), str(out_base)):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Baigta! Sėkmingai: {success}, Nepavyko: {failed}")


def main():
    """Pagrindinis CLI įėjimo taškas."""
    parser = argparse.ArgumentParser(
        description="Vessel Auto Measure CLI - Akies dugno kraujagyslių analizė",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pavyzdžiai:
  python cli.py -i /kelias/iki/images -o /kelias/iki/results
  python cli.py -i image.jpg -o result
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Įvesties failas arba aplankas'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='Išvesties failas arba aplankas (numatytasis: output)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"KLAIDA: Nerastas kelias: {args.input}")
        sys.exit(1)

    if input_path.is_dir():
        process_directory(args.input, args.output)
    else:
        # Vienas failas, jei -o be plėtinio, traktuoti kaip aplanką
        output = args.output
        if output.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            output_base = output
        else:
            Path(output).mkdir(parents=True, exist_ok=True)
            output_base = str(Path(output) / input_path.stem)

        success = process_single_image(args.input, output_base)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()