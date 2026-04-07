#!/usr/bin/env python3
"""
Script to analyze SODA-A dataset structure and statistics.

This script provides:
- Dataset structure analysis
- Class distribution statistics
- Ship-related class extraction
- Resolution estimation
- Export statistics to JSON

Usage:
    python scripts/analyze_dataset.py --data_root /path/to/soda-a --output stats.json
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_analysis import SODAAAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SODA-A dataset structure and statistics"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to SODA-A dataset root directory"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/dataset_stats.json",
        help="Output path for statistics JSON file"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SODA-A Dataset Analyzer")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SODAAAnalyzer(args.data_root)
    
    # Run analysis
    print(f"\nAnalyzing dataset at: {args.data_root}")
    analyzer.analyze_dataset(verbose=not args.quiet)
    
    # Print summary
    analyzer.print_summary()
    
    # Export statistics
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    analyzer.export_statistics(args.output)
    
    print(f"\nStatistics exported to: {args.output}")


if __name__ == "__main__":
    main()
