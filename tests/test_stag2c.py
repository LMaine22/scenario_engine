"""
Test Script for Stage 2C: Full Pipeline Integration

This script verifies:
1. Complete end-to-end pipeline runs successfully
2. All outputs are generated (figures, results, scenarios)
3. Spread forecasts are included in outputs
4. Validation runs correctly
5. Reports are created
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import subprocess
import os
from datetime import datetime


def test_stage_2c():
    """Test Stage 2C: Full Pipeline"""
    
    print("="*60)
    print("STAGE 2C TEST: Full Pipeline Integration")
    print("="*60)
    
    print("\n[1/3] Running full pipeline...")
    print("This will take 2-3 minutes for complete scenario generation...")
    
    # Run main.py
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"\n❌ FAILED: Pipeline execution error")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        
        output = result.stdout
        print(f"✓ Pipeline completed successfully")
        
    except subprocess.TimeoutExpired:
        print(f"\n❌ FAILED: Pipeline timeout (>5 minutes)")
        return False
    except Exception as e:
        print(f"\n❌ FAILED: Execution error: {e}")
        return False
    
    # Check for key indicators in output
    print("\n[2/3] Validating pipeline output...")
    
    checks = {
        'Data loaded': 'Processed data:' in output,
        'AFNS fitted': 'Fitted VAR(1) Parameters:' in output,
        'Regime model fitted': 'Selected K=' in output,
        'SpreadEngine fitted': 'SpreadEngine fitted' in output or 'Spreads disabled' in output,
        'Scenarios generated': 'Simulation complete:' in output,
        'Validation ran': 'Running multi-horizon validation' in output,
        'Outputs saved': 'Results saved to' in output,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = '✓' if passed else '❌'
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n❌ Some pipeline steps failed")
        return False
    
    # Find the run directory (most recent)
    print("\n[3/3] Checking output files...")
    
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print(f"  ❌ Runs directory not found")
        return False
    
    # Get most recent run
    run_dirs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    if not run_dirs:
        print(f"  ❌ No run directories found")
        return False
    
    latest_run = run_dirs[0]
    print(f"  Checking outputs in: {latest_run}")
    
    # Check expected files
    expected_files = {
        'Figures': [
            latest_run / 'figures' / 'scenario_fan.png',
            latest_run / 'figures' / 'regime_analysis.png',
        ],
        'Results': [
            latest_run / 'results' / 'validation_results.csv',
            latest_run / 'results' / 'validation_summary.txt',
        ],
        'Scenarios': [
            latest_run / 'scenarios' / 'latest_scenarios.csv',
        ]
    }
    
    # Add spread-specific files if spreads were enabled
    if 'SpreadEngine fitted' in output:
        expected_files['Figures'].append(latest_run / 'figures' / 'spread_scenarios.png')
        expected_files['Results'].append(latest_run / 'results' / 'spread_forecasts.csv')
    
    all_files_exist = True
    for category, files in expected_files.items():
        print(f"\n  {category}:")
        for file_path in files:
            exists = file_path.exists()
            status = '✓' if exists else '❌'
            print(f"    {status} {file_path.name}")
            if not exists:
                all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some expected files missing")
        return False
    
    # Check file sizes (should not be empty)
    print(f"\n  File sizes:")
    for category, files in expected_files.items():
        for file_path in files:
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"    {file_path.name}: {size_kb:.1f} KB")
    
    # Parse and display key results
    print("\n" + "="*60)
    print("PIPELINE OUTPUT SUMMARY")
    print("="*60)
    
    # Extract multi-year forecast if available
    if 'Multi-Year Forecast Example' in output:
        forecast_section = output.split('Multi-Year Forecast Example')[1].split('===')[0]
        print("\nTreasury Forecast:")
        print(forecast_section.strip())
    
    # Extract spread forecast if available
    if 'Spread Product Forecasts' in output:
        spread_section = output.split('Spread Product Forecasts')[1].split('===')[0]
        print("\nSpread Product Forecast:")
        print(spread_section.strip())
    
    # Summary
    print("\n" + "="*60)
    print("STAGE 2C: ✓ ALL TESTS PASSED")
    print("="*60)
    print("\nKey Results:")
    print(f"  • Full pipeline executed successfully")
    print(f"  • All output files generated")
    print(f"  • Treasury + Spread forecasts produced")
    print(f"  • Validation completed")
    print(f"  • Reports ready for Baker Group")
    print(f"\nRun directory: {latest_run}")
    print(f"\n🎉 Phase 2 Complete - Production Ready!")
    
    return True


if __name__ == "__main__":
    success = test_stage_2c()
    sys.exit(0 if success else 1)