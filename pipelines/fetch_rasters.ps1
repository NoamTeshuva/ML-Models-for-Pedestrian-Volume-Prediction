param(
    [string[]]$Cities = @("melbourne", "NewYork", "zurich", "dublin"),
    [int]$Year = 2023,
    [switch]$Smoke,        # Use *_cbd AOI for quick test
    [switch]$SkipDEM,      # Skip DEM fetching
    [switch]$SkipNDVI,     # Skip NDVI fetching  
    [switch]$Verbose,      # Enable verbose logging
    [switch]$DryRun        # Show commands without executing
)

# STAC Raster Fetcher Pipeline
# Automated downloading of city-scale DEM and NDVI rasters from cloud catalogs

Write-Host "=== STAC Raster Fetcher Pipeline ===" -ForegroundColor Magenta
Write-Host "Cities: $($Cities -join ', ')" -ForegroundColor Cyan
Write-Host "Year: $Year" -ForegroundColor Cyan

if ($Smoke) {
    Write-Host "Mode: SMOKE TEST (CBD areas only)" -ForegroundColor Yellow
} else {
    Write-Host "Mode: FULL CITY" -ForegroundColor Green
}

if ($SkipDEM) {
    Write-Host "Skipping: DEM fetching" -ForegroundColor Yellow
}

if ($SkipNDVI) {
    Write-Host "Skipping: NDVI fetching" -ForegroundColor Yellow
}

Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python and add to PATH." -ForegroundColor Red
    exit 1
}

# Install/update dependencies if needed
Write-Host "Installing/updating STAC fetch dependencies..." -ForegroundColor Cyan
if (-not $DryRun) {
    try {
        pip install -r scripts\requirements_fetch.txt --quiet --upgrade
        Write-Host "Dependencies installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Dependency installation failed. Proceeding anyway..." -ForegroundColor Yellow
        Write-Host "Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[DRY RUN] Would run: pip install -r scripts\requirements_fetch.txt --quiet --upgrade"
}

Write-Host ""

# Track overall results
$results = @()
$successCount = 0
$failureCount = 0

# Process each city
foreach ($city in $Cities) {
    $preset = if ($Smoke) { "${city}_cbd" } else { $city }
    
    Write-Host "Processing $city (AOI: $preset)..." -ForegroundColor Green
    Write-Host "=================================================================================="
    
    # Build command arguments
    $args = @(
        "scripts\get_rasters_stac.py",
        "--city", $city,
        "--year", $Year,
        "--aoi-preset", $preset
    )
    
    # Add conditional arguments
    if ($SkipDEM) {
        $args += "--ndvi-only"
    }
    
    if ($SkipNDVI) {
        $args += "--dem-only"  
    }
    
    if ($Verbose) {
        $args += "--verbose"
    }
    
    if ($Smoke) {
        $args += "--smoke-test"
    }
    
    # Display command
    $cmdDisplay = "python " + ($args -join " ")
    Write-Host "Command: $cmdDisplay" -ForegroundColor Gray
    Write-Host ""
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would execute above command" -ForegroundColor Yellow
        $results += [PSCustomObject]@{
            City = $city
            Preset = $preset
            Status = "DRY_RUN"
            Command = $cmdDisplay
        }
        continue
    }
    
    # Execute the fetch command
    $startTime = Get-Date
    try {
        $output = python @args 2>&1
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($exitCode -eq 0) {
            Write-Host "SUCCESS: $city completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
            $successCount++
            $status = "SUCCESS"
        } else {
            Write-Host "FAILED: $city failed with exit code $exitCode" -ForegroundColor Red
            Write-Host "Output: $output" -ForegroundColor Red
            $failureCount++
            $status = "FAILED"
        }
        
        $results += [PSCustomObject]@{
            City = $city
            Preset = $preset
            Status = $status
            ExitCode = $exitCode
            Duration = $duration.TotalMinutes.ToString('F1') + " min"
            Command = $cmdDisplay
        }
        
    } catch {
        Write-Host "ERROR: Exception occurred while processing $city" -ForegroundColor Red
        Write-Host "Exception: $_" -ForegroundColor Red
        $failureCount++
        
        $results += [PSCustomObject]@{
            City = $city
            Preset = $preset
            Status = "ERROR"
            ExitCode = -1
            Duration = "N/A"
            Command = $cmdDisplay
            Error = $_.Exception.Message
        }
    }
    
    Write-Host ""
}

# Print summary report
Write-Host "=== PIPELINE SUMMARY ===" -ForegroundColor Magenta
Write-Host "Cities processed: $($Cities.Count)" -ForegroundColor Cyan
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Failed: $failureCount" -ForegroundColor Red

if ($failureCount -eq 0 -and -not $DryRun) {
    Write-Host "All cities completed successfully!" -ForegroundColor Green
} elseif ($failureCount -gt 0 -and -not $DryRun) {
    Write-Host "Some cities failed - check logs above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Results by City:" -ForegroundColor Cyan
$results | Format-Table City, Preset, Status, Duration, ExitCode -AutoSize

# Show output file locations
if (-not $DryRun -and $successCount -gt 0) {
    Write-Host ""
    Write-Host "Output files created:" -ForegroundColor Cyan
    
    foreach ($result in $results) {
        if ($result.Status -eq "SUCCESS") {
            $city = $result.City
            $demPath = "data\external\$city\dem.tif"
            $ndviPath = "data\external\$city\green.tif"
            
            if (Test-Path $demPath) {
                $demSize = [math]::Round((Get-Item $demPath).Length / 1MB, 1)
                Write-Host "  $demPath ($demSize MB)" -ForegroundColor Green
            }
            
            if (Test-Path $ndviPath) {
                $ndviSize = [math]::Round((Get-Item $ndviPath).Length / 1MB, 1)
                Write-Host "  $ndviPath ($ndviSize MB)" -ForegroundColor Green
            }
        }
    }
}

# Show latest reports
$latestReports = Get-ChildItem -Path "reports\runs" -Filter "*_fetch_summary.md" -File | 
                 Sort-Object LastWriteTime -Descending | 
                 Select-Object -First 3

if ($latestReports) {
    Write-Host ""
    Write-Host "Latest fetch reports:" -ForegroundColor Cyan
    foreach ($report in $latestReports) {
        Write-Host "  $($report.FullName)" -ForegroundColor Gray
    }
}

# Suggest next steps
if (-not $DryRun -and $successCount -gt 0) {
    Write-Host ""
    Write-Host "=== NEXT STEPS ===" -ForegroundColor Magenta
    Write-Host "1. Verify raster quality:" -ForegroundColor Yellow
    Write-Host "   python scripts\verify_external_rasters.py --cities $($Cities -join ' ')" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Run feature extraction pipeline:" -ForegroundColor Yellow
    Write-Host "   python pipelines\step1_melbourne_edge_env_v3.py --city melbourne --year $Year" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Check progress:" -ForegroundColor Yellow  
    Write-Host "   python scripts\verify_progress.py --cities $($Cities -join ' ') --year $Year" -ForegroundColor Gray
}

# Exit with appropriate code
if ($DryRun) {
    Write-Host "Dry run completed" -ForegroundColor Blue
    exit 0
} elseif ($failureCount -gt 0) {
    Write-Host "Pipeline completed with failures" -ForegroundColor Red
    exit 1
} else {
    Write-Host "Pipeline completed successfully" -ForegroundColor Green
    exit 0
}