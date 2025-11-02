param(
    [string[]]$Cities = @("melbourne", "NewYork", "zurich", "dublin"), 
    [int]$Year = 2023
)

Write-Host "Auditing raster fetch capability..." -ForegroundColor Cyan

# Convert Cities array to space-separated string for Python argparse
$CitiesStr = $Cities -join " "

# Run the Python audit script
python scripts\audit_raster_fetchers.py --cities $Cities --year $Year

if ($LASTEXITCODE -ne 0) {
    Write-Host "Audit script failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Find the most recent audit report
$md = Get-ChildItem -Recurse -File "reports\runs\*_raster_fetch_audit.md" | 
      Sort-Object LastWriteTime -Descending | 
      Select-Object -First 1

if ($md) {
    Write-Host "Audit report:" $md.FullName -ForegroundColor Green
    
    # Also find the JSON report
    $json = Get-ChildItem -Recurse -File "reports\metrics\raster_fetch_audit_*.json" | 
            Sort-Object LastWriteTime -Descending | 
            Select-Object -First 1
    
    if ($json) {
        Write-Host "JSON metrics:" $json.FullName -ForegroundColor Green
    }
    
    Write-Host "`nRun 'Get-Content `"$($md.FullName)`"' to view the full report." -ForegroundColor Yellow
} else {
    Write-Host "No audit report found." -ForegroundColor Yellow
}