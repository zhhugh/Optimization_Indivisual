#!/usr/bin/env python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.generate_tables import generate_all_tables
from src.analysis.generate_figures import generate_all_figures


def main():
    """Generate all analysis outputs."""
    print("=" * 70)
    print("GENERATING ALL ANALYSIS OUTPUTS")
    print("=" * 70)

    # Generate tables
    print("\n" + "=" * 70)
    print("STEP 1: GENERATING TABLES")
    print("=" * 70)
    try:
        generate_all_tables(log_dir="experiments/logs")
    except Exception as e:
        print(f"Error generating tables: {e}")
        import traceback
        traceback.print_exc()

    # Generate figures
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING FIGURES")
    print("=" * 70)
    try:
        generate_all_figures(log_dir="experiments/logs", save_dir="reports/figures")
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
