#Requires -Version 5.0
<#
.SYNOPSIS
    Orchestrate full environmental pipeline for multiple cities with preflight checks and auto-tuning.

.DESCRIPTION
    Runs the complete environmental features pipeline (fetch rasters, extract edge features, 
    merge with sensors) for specified cities using the new preflight coverage checks and 
    auto-tuning capabilities. Tracks timing and generates QA reports.

.PARAMETER Cities
    Comma-separated list of cities to process (e.g., "melbourne,NewYork")

.PARAMETER Year
    Year for processing (e.g., 2023)

.PARAMETER Smoke
    Run in smoke test mode (faster, smaller datasets)

.PARAMETER VerboseOutput
    Enable verbose logging

.EXAMPLE
    .\run_env_update_2023.ps1 -Cities "melbourne,NewYork" -Year 2023 -VerboseOutput

.NOTES
    This script integrates the new preflight checks and auto-tuning features:
    - --preflight-min-coverage 0.30 for both DEM and NDVI
    - --auto-ndvi-threshold for intelligent canopy thresholds
    - --auto-adjust-seg for optimized topography processing
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Cities,
    
    [Parameter(Mandatory=$true)]
    [int]$Year,
    
    [switch]$Smoke = $false,
    
    [switch]$VerboseOutput = $false
)

# Set error handling
$ErrorActionPreference = "Continue"  # Continue to next city on errors
$VerbosePreference = if ($VerboseOutput) { "Continue" } else { "SilentlyContinue" }

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host "Environmental Pipeline Orchestrator - Year $Year" -ForegroundColor Cyan
Write-Host "Cities: $Cities" -ForegroundColor White
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host ""

# Parse cities list
$CityList = $Cities -split "," | ForEach-Object { $_.Trim() }
Write-Host "Processing $($CityList.Count) cities: $($CityList -join ', ')" -ForegroundColor Green

# Initialize timing and results tracking
$Results = @{}
$OverallStartTime = Get-Date

# Create necessary directories
$ReportsDir = Join-Path $ProjectRoot "reports"
$MetricsDir = Join-Path $ReportsDir "metrics"
$RunsDir = Join-Path $ReportsDir "runs"

@($ReportsDir, $MetricsDir, $RunsDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -Path $_ -ItemType Directory -Force | Out-Null
        Write-Verbose "Created directory: $_"
    }
}

# Function to measure execution time
function Invoke-TimedCommand {
    param(
        [string]$Name,
        [scriptblock]$Command,
        [string]$City = "global"
    )
    
    Write-Host "STARTING: $Name" -ForegroundColor Yellow
    $StartTime = Get-Date
    
    try {
        $Output = & $Command
        $Success = $LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        
        if ($Success) {
            Write-Host "SUCCESS: Completed: $Name ($([math]::Round($Duration, 1))s)" -ForegroundColor Green
        } else {
            Write-Host "FAILED: $Name ($([math]::Round($Duration, 1))s) - Exit Code: $LASTEXITCODE" -ForegroundColor Red
        }
        
        # Track timing
        if (-not $Results.ContainsKey($City)) {
            $Results[$City] = @{ Stages = @{}; Success = $true }
        }
        $Results[$City].Stages[$Name] = @{
            Duration = $Duration
            Success = $Success
            ExitCode = $LASTEXITCODE
            StartTime = $StartTime.ToString("yyyy-MM-dd HH:mm:ss")
            EndTime = $EndTime.ToString("yyyy-MM-dd HH:mm:ss")
        }
        
        if (-not $Success) {
            $Results[$City].Success = $false
        }
        
        return @{ Success = $Success; Duration = $Duration; Output = $Output }
        
    } catch {
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds
        Write-Host "EXCEPTION in: $Name - $($_.Exception.Message)" -ForegroundColor Red
        
        $Results[$City].Stages[$Name] = @{
            Duration = $Duration
            Success = $false
            Error = $_.Exception.Message
            StartTime = $StartTime.ToString("yyyy-MM-dd HH:mm:ss")
            EndTime = $EndTime.ToString("yyyy-MM-dd HH:mm:ss")
        }
        $Results[$City].Success = $false
        
        return @{ Success = $false; Duration = $Duration; Error = $_.Exception.Message }
    }
}

# Function to save timing metrics
function Save-TimingMetrics {
    param([string]$City, [hashtable]$CityResults)
    
    $TimingFile = Join-Path $MetricsDir "timing_pipeline_${City}_${Year}.json"
    $TimingData = @{
        city = $City
        year = $Year
        timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        success = $CityResults.Success
        stages = $CityResults.Stages
        total_duration = ($CityResults.Stages.Values | Measure-Object -Property Duration -Sum).Sum
    }
    
    $TimingData | ConvertTo-Json -Depth 5 | Out-File -FilePath $TimingFile -Encoding UTF8
    Write-Verbose "Saved timing metrics: $TimingFile"
}

# Process each city
foreach ($City in $CityList) {
    Write-Host ""
    Write-Host "  Processing City: $City" -ForegroundColor Magenta
    Write-Host "=" * 50
    
    $CityStartTime = Get-Date
    
    # Stage 1: Fetch Rasters
    Write-Host ""
    Write-Host " Stage 1: Fetching Rasters" -ForegroundColor Blue
    $FetchArgs = @("-Cities", $City, "-Year", $Year)
    if ($Smoke) { $FetchArgs += "-Smoke" }
    
    $FetchResult = Invoke-TimedCommand -Name "fetch_rasters" -City $City -Command {
        & powershell -ExecutionPolicy Bypass -File "pipelines\fetch_rasters.ps1" @FetchArgs
    }
    
    if (-not $FetchResult.Success) {
        Write-Host "  Raster fetch failed for $City, but continuing..." -ForegroundColor Yellow
    }
    
    # Stage 2: Edge Environmental Features (with preflight and auto-tuning)
    Write-Host ""
    Write-Host " Stage 2: Edge Environmental Features (with guardrails)" -ForegroundColor Blue
    
    $EdgeResult = Invoke-TimedCommand -Name "edge_env_features" -City $City -Command {
        python "pipelines\step1_melbourne_edge_env_v3.py" --city $City --year $Year
    }
    
    if (-not $EdgeResult.Success) {
        Write-Host " Edge feature extraction failed for $City" -ForegroundColor Red
        Write-Host "This could be due to preflight coverage checks failing" -ForegroundColor Yellow
        Write-Host "Check that DEM and NDVI rasters have sufficient coverage for this city" -ForegroundColor Yellow
        continue  # Skip to next city
    }
    
    # Stage 3: Sensor Merge
    Write-Host ""
    Write-Host " Stage 3: Sensor Environmental Merge" -ForegroundColor Blue
    
    $MergeResult = Invoke-TimedCommand -Name "sensor_merge" -City $City -Command {
        python "pipelines\step2_melbourne_sensor_merge_v3.py" --city $City --year $Year
    }
    
    if (-not $MergeResult.Success) {
        Write-Host " Sensor merge failed for $City" -ForegroundColor Red
        continue
    }
    
    # Calculate city total time
    $CityEndTime = Get-Date
    $CityTotalTime = ($CityEndTime - $CityStartTime).TotalSeconds
    
    Write-Host ""
    Write-Host " City $City completed in $([math]::Round($CityTotalTime, 1)) seconds" -ForegroundColor Green
    
    # Save timing metrics for this city
    Save-TimingMetrics -City $City -CityResults $Results[$City]
}

# Overall summary
Write-Host ""
Write-Host " Pipeline Summary" -ForegroundColor Cyan
Write-Host "=" * 50

$SuccessfulCities = $Results.Keys | Where-Object { $Results[$_].Success }
$FailedCities = $Results.Keys | Where-Object { -not $Results[$_].Success }

Write-Host " Successful cities: $($SuccessfulCities.Count)/$($CityList.Count)" -ForegroundColor Green
if ($SuccessfulCities.Count -gt 0) {
    $SuccessfulCities | ForEach-Object { Write-Host "   - $_" -ForegroundColor Green }
}

if ($FailedCities.Count -gt 0) {
    Write-Host " Failed cities: $($FailedCities.Count)" -ForegroundColor Red
    $FailedCities | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
}

# Calculate overall timing
$OverallEndTime = Get-Date
$OverallDuration = ($OverallEndTime - $OverallStartTime).TotalSeconds
Write-Host ""
Write-Host "  Total pipeline time: $([math]::Round($OverallDuration, 1)) seconds" -ForegroundColor White

# Run progress verification
Write-Host ""
Write-Host " Running Progress Verification" -ForegroundColor Blue

$VerifyResult = Invoke-TimedCommand -Name "verify_progress" -Command {
    python "scripts\verify_progress.py" --cities $Cities --year $Year
}

if ($VerifyResult.Success) {
    Write-Host " Progress verification passed" -ForegroundColor Green
} else {
    Write-Host " Progress verification failed" -ForegroundColor Red
}

# Final status
Write-Host ""
Write-Host " Pipeline Complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run QA signal report: python scripts\qa_env_signal_report.py --cities $Cities --year $Year" -ForegroundColor White
Write-Host "2. Check timing metrics in: reports\metrics\timing_*.json" -ForegroundColor White
Write-Host "3. Review city outputs in: data\processed\<city>\csv\" -ForegroundColor White

# Exit with appropriate code
if ($FailedCities.Count -eq 0) {
    Write-Host " All cities processed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "  Some cities failed processing" -ForegroundColor Yellow  
    exit 1
}
