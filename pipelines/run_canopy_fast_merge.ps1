# PowerShell orchestrator for fast canopy merge and inspection
# Merges FAST sensor features into base table and inspects the results

Write-Host "=== Fast Canopy Merge Pipeline ===" -ForegroundColor Green
Write-Host ""

# Auto-locate base feature table
Write-Host "Locating base feature table..." -ForegroundColor Cyan
$basePattern = "data\processed\melbourne\csv\feature_table_2023_melbourne.csv"

# Find files matching pattern but exclude those with "_with_" in name
$candidateFiles = Get-ChildItem -Path $basePattern -ErrorAction SilentlyContinue | Where-Object { 
    $_.Name -notlike "*_with_*" 
}

if (-not $candidateFiles) {
    Write-Host "ERROR: No base feature table found matching pattern: $basePattern" -ForegroundColor Red
    Write-Host "Looked for: feature_table_2023_melbourne.csv (excluding files with '_with_' in name)" -ForegroundColor Red
    exit 1
}

$baseFile = $candidateFiles[0].FullName
Write-Host "Found base table: $baseFile"

# Set sensor features path
$sensFile = "data\processed\melbourne\csv\sensor_env_features_2023_melbourne_FAST_FIX.csv"
Write-Host "Sensor features: $sensFile"

# Check if sensor features file exists
if (-not (Test-Path $sensFile)) {
    Write-Host "ERROR: Sensor features file not found: $sensFile" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Run merge script
Write-Host "--- Step 1: Merging canopy features ---" -ForegroundColor Cyan
Write-Host "Running: python scripts\merge_canopy_fast_fix.py"
Write-Host ""

& python scripts\merge_canopy_fast_fix.py --base $baseFile --sens $sensFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Merge failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""

# Construct expected output path (same logic as merge script)
$basePathObj = Get-Item $baseFile
$baseStem = $basePathObj.BaseName
$baseExt = $basePathObj.Extension
$expectedOutput = Join-Path $basePathObj.DirectoryName "${baseStem}_with_canopyFAST_FIX${baseExt}"

# Check if the expected output exists, or look for versioned files
$actualOutput = $expectedOutput
if (-not (Test-Path $expectedOutput)) {
    # Look for versioned files
    $versionPattern = Join-Path $basePathObj.DirectoryName "${baseStem}_with_canopyFAST_FIX__v*${baseExt}"
    $versionedFiles = Get-ChildItem -Path $versionPattern -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    
    if ($versionedFiles) {
        $actualOutput = $versionedFiles[0].FullName
    }
}

if (-not (Test-Path $actualOutput)) {
    Write-Host "ERROR: Expected output file not found: $actualOutput" -ForegroundColor Red
    exit 1
}

Write-Host "Merged file created: $actualOutput"

# Run inspection
Write-Host "--- Step 2: Inspecting merge results ---" -ForegroundColor Cyan
Write-Host "Running: python scripts\inspect_canopy_merge.py"
Write-Host ""

& python scripts\inspect_canopy_merge.py --file $actualOutput

Write-Host ""

# Read the merged file to extract key statistics for summary
Write-Host "--- Pipeline Summary ---" -ForegroundColor Green

try {
    # Use Python to quickly extract key stats
    $statsScript = @"
import pandas as pd
import sys

try:
    df = pd.read_csv('$($actualOutput.Replace('\', '\\'))')
    canopy_cols = ['sensor_canopy_pct', 'sensor_ndvi_mean', 'sensor_canopy_valid_frac']
    existing_cols = [col for col in canopy_cols if col in df.columns]
    
    print(f"Rows in merged file: {len(df):,}")
    
    for col in existing_cols:
        non_null_pct = (df[col].notna().sum() / len(df)) * 100
        print(f"{col}: {non_null_pct:.1f}% non-null")
    
    if 'sensor_canopy_pct' in df.columns:
        mean_canopy = df['sensor_canopy_pct'].mean()
        if pd.notna(mean_canopy):
            print(f"Average canopy coverage: {mean_canopy:.1%}")
            
except Exception as e:
    print(f"Could not extract summary stats: {e}")
"@

    $statsOutput = python -c $statsScript
    Write-Host $statsOutput
    
} catch {
    Write-Host "Could not extract detailed statistics from merged file" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Files used:" -ForegroundColor White
Write-Host "  Base table: $baseFile"
Write-Host "  Sensor features: $sensFile" 
Write-Host "  Merged output: $actualOutput"

Write-Host ""
Write-Host "Fast canopy merge pipeline completed successfully!" -ForegroundColor Green