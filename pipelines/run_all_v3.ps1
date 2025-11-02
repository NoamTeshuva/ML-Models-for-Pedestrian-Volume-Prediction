# v3 Rollout Orchestrator - Complete Pipeline Runner
# 
# Runs complete v3 validation and rollout process across configured cities.
# Uses timing instrumentation to track performance and generates comprehensive summary.

param(
    [string[]] $Cities = @("melbourne", "NewYork", "zurich", "dublin"),
    [int] $Year = 2023,
    [string] $Config = "config/cities.yaml",
    [switch] $Verbose,
    [switch] $Force,
    [switch] $SkipComparison,
    [switch] $SkipABEval,
    [switch] $DryRun
)

# Script configuration
$ErrorActionPreference = "Continue"  # Continue processing other cities on failures
$ProgressPreference = "Continue"

# Ensure Python is available
try {
    python --version | Out-Null
} catch {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Create reports directories
$ReportsDir = "reports"
$MetricsDir = "reports/metrics"
$RunsDir = "reports/runs"

@($ReportsDir, $MetricsDir, $RunsDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}

$StartTime = Get-Date
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "v3 ROLLOUT ORCHESTRATOR" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cities: $($Cities -join ', ')" -ForegroundColor White
Write-Host "Year: $Year" -ForegroundColor White
Write-Host "Config: $Config" -ForegroundColor White
Write-Host "Started: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
Write-Host ""

# Function to execute timed command
function Invoke-TimedCommand {
    param(
        [string]$Stage,
        [string]$City,
        [int]$Year,
        [string[]]$Command,
        [hashtable]$Meta = @{}
    )
    
    $MetaJson = ($Meta | ConvertTo-Json -Compress).Replace('"', '\"')
    $OutputFile = "reports/metrics/timing_${Stage}_${City}_${Year}.json"
    
    $TimingCmd = @(
        "python", "pipelines/track_time.py",
        "--stage", $Stage,
        "--meta", $MetaJson,
        "--out-json", $OutputFile,
        "--"
    ) + $Command
    
    if ($DryRun) {
        Write-Host "[DRY RUN] Would execute: $($TimingCmd -join ' ')" -ForegroundColor Yellow
        return @{ Success = $true; ExitCode = 0; Output = "DRY RUN" }
    }
    
    if ($Verbose) {
        Write-Host "Executing: $($TimingCmd -join ' ')" -ForegroundColor Gray
    }
    
    # Execute with timing
    $Process = Start-Process -FilePath "python" -ArgumentList ($TimingCmd[1..($TimingCmd.Length-1)]) -Wait -PassThru -NoNewWindow
    
    $Result = @{
        Success = $Process.ExitCode -eq 0
        ExitCode = $Process.ExitCode
        TimingFile = $OutputFile
    }
    
    # Read timing information
    if (Test-Path $OutputFile) {
        try {
            $TimingData = Get-Content $OutputFile | ConvertFrom-Json
            $Result['Duration'] = $TimingData.duration_seconds
            $Result['StartTime'] = $TimingData.start_time
            $Result['EndTime'] = $TimingData.end_time
        } catch {
            Write-Warning "Could not read timing data from $OutputFile"
        }
    }
    
    return $Result
}

# Function to check if required files exist for city
function Test-CityDataAvailability {
    param([string]$City, [int]$Year)
    
    # Normalize city name for file paths
    $NormalizedCity = switch ($City.ToLower()) {
        'newyork' { 'NewYork' }
        'new_york' { 'NewYork' }
        default { $City }
    }
    
    $RequiredFiles = @(
        "data/processed/$NormalizedCity/gpkg/street_network_${Year}_${City}.gpkg",
        "data/processed/$NormalizedCity/raster/dem_${Year}_${City}.tif",
        "data/processed/$NormalizedCity/raster/green_${Year}_${City}.tif"
    )
    
    $MissingFiles = @()
    foreach ($File in $RequiredFiles) {
        if (-not (Test-Path $File)) {
            # Try alternative path structure
            $AltFile = $File.Replace($NormalizedCity, $City)
            if (-not (Test-Path $AltFile)) {
                $MissingFiles += $File
            }
        }
    }
    
    return @{
        Available = $MissingFiles.Count -eq 0
        MissingFiles = $MissingFiles
    }
}

# Initialize results tracking
$CityResults = @{}
$OverallSuccess = $true

# Process each city
foreach ($City in $Cities) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "PROCESSING: $($City.ToUpper()) ($Year)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    $CityStartTime = Get-Date
    $CityResults[$City] = @{
        City = $City
        Year = $Year
        StartTime = $CityStartTime
        Steps = @{}
        Success = $false
    }
    
    # Check data availability
    $DataCheck = Test-CityDataAvailability -City $City -Year $Year
    if (-not $DataCheck.Available -and -not $Force) {
        Write-Host "Missing required data files for $City" -ForegroundColor Red
        foreach ($File in $DataCheck.MissingFiles) {
            Write-Host "  ❌ $File" -ForegroundColor Red
        }
        Write-Host "Use -Force to skip data checks" -ForegroundColor Yellow
        $CityResults[$City]['Error'] = "Missing required data files"
        $CityResults[$City]['MissingFiles'] = $DataCheck.MissingFiles
        continue
    } elseif (-not $DataCheck.Available) {
        Write-Host "⚠️ Missing data files but continuing with -Force flag" -ForegroundColor Yellow
    }
    
    try {
        # Step 1: v3 Environmental Extraction
        Write-Host "Step 1: v3 Environmental Feature Extraction" -ForegroundColor Cyan
        
        $EnvCommand = @(
            "python", "pipelines/run_env_v3.py",
            "--city", $City,
            "--year", $Year.ToString(),
            "--force-v3"
        )
        
        if ($Verbose) { $EnvCommand += "--verbose" }
        
        $EnvResult = Invoke-TimedCommand -Stage "env_v3" -City $City -Year $Year -Command $EnvCommand -Meta @{city=$City; year=$Year}
        $CityResults[$City]['Steps']['env_v3'] = $EnvResult
        
        if ($EnvResult.Success) {
            Write-Host "  ✅ v3 extraction completed in $($EnvResult.Duration)s" -ForegroundColor Green
        } else {
            Write-Host "  ❌ v3 extraction failed (exit code: $($EnvResult.ExitCode))" -ForegroundColor Red
            $OverallSuccess = $false
            continue
        }
        
        # Step 2: Sensor Pipeline Integration
        Write-Host "Step 2: Sensor Pipeline Integration" -ForegroundColor Cyan
        
        # Look for existing sensor pipeline script
        $SensorScripts = @(
            "scripts/run_sensor_env_pipeline.ps1",
            "pipelines/run_sensor_env_pipeline.ps1",
            "src/data_processing/run_sensor_pipeline.py"
        )
        
        $SensorScript = $null
        foreach ($Script in $SensorScripts) {
            if (Test-Path $Script) {
                $SensorScript = $Script
                break
            }
        }
        
        if ($SensorScript) {
            if ($SensorScript.EndsWith('.ps1')) {
                $SensorCommand = @("powershell", "-ExecutionPolicy", "Bypass", "-File", $SensorScript, $City, $Year.ToString())
            } else {
                $SensorCommand = @("python", $SensorScript, $City, $Year.ToString())
            }
            
            $SensorResult = Invoke-TimedCommand -Stage "sensor_merge" -City $City -Year $Year -Command $SensorCommand
            $CityResults[$City]['Steps']['sensor_merge'] = $SensorResult
            
            if ($SensorResult.Success) {
                Write-Host "  ✅ Sensor integration completed in $($SensorResult.Duration)s" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️ Sensor integration failed (exit code: $($SensorResult.ExitCode))" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ⚠️ No sensor pipeline script found - skipping integration" -ForegroundColor Yellow
            $CityResults[$City]['Steps']['sensor_merge'] = @{ Success = $false; Skipped = $true; Reason = "Script not found" }
        }
        
        # Step 3: v2 vs v3 Comparison (optional)
        if (-not $SkipComparison) {
            Write-Host "Step 3: v2 vs v3 Feature Comparison" -ForegroundColor Cyan
            
            $CompareCommand = @(
                "python", "pipelines/compare_v2_v3_env.py",
                "--city", $City,
                "--year", $Year.ToString()
            )
            
            if ($Verbose) { $CompareCommand += "--verbose" }
            
            $CompareResult = Invoke-TimedCommand -Stage "compare_v2_v3" -City $City -Year $Year -Command $CompareCommand
            $CityResults[$City]['Steps']['compare_v2_v3'] = $CompareResult
            
            if ($CompareResult.Success) {
                Write-Host "  ✅ Comparison completed in $($CompareResult.Duration)s" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️ Comparison failed or no v2 data available (exit code: $($CompareResult.ExitCode))" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ⚪ Skipping v2 vs v3 comparison" -ForegroundColor Gray
            $CityResults[$City]['Steps']['compare_v2_v3'] = @{ Success = $false; Skipped = $true; Reason = "User requested skip" }
        }
        
        # Step 4: A/B Model Evaluation (optional)
        if (-not $SkipABEval) {
            Write-Host "Step 4: A/B Model Evaluation" -ForegroundColor Cyan
            
            $ABCommand = @(
                "python", "pipelines/run_ab_model_eval.py",
                "--city", $City,
                "--year", $Year.ToString(),
                "--cv-folds", "3"
            )
            
            if ($Verbose) { $ABCommand += "--verbose" }
            
            $ABResult = Invoke-TimedCommand -Stage "ab_eval" -City $City -Year $Year -Command $ABCommand
            $CityResults[$City]['Steps']['ab_eval'] = $ABResult
            
            if ($ABResult.Success) {
                Write-Host "  ✅ A/B evaluation completed in $($ABResult.Duration)s" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️ A/B evaluation failed or no training data available (exit code: $($ABResult.ExitCode))" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ⚪ Skipping A/B model evaluation" -ForegroundColor Gray
            $CityResults[$City]['Steps']['ab_eval'] = @{ Success = $false; Skipped = $true; Reason = "User requested skip" }
        }
        
        # Determine city success (require at least env_v3 success)
        $CityResults[$City]['Success'] = $EnvResult.Success
        $CityEndTime = Get-Date
        $CityResults[$City]['EndTime'] = $CityEndTime
        $CityResults[$City]['Duration'] = ($CityEndTime - $CityStartTime).TotalSeconds
        
        if ($CityResults[$City]['Success']) {
            Write-Host "✅ $City processing completed successfully in $([math]::Round($CityResults[$City]['Duration'], 1))s" -ForegroundColor Green
        } else {
            Write-Host "❌ $City processing failed" -ForegroundColor Red
        }
        
    } catch {
        Write-Host "❌ Unexpected error processing ${City}: $($_.Exception.Message)" -ForegroundColor Red
        $CityResults[$City]['Error'] = $_.Exception.Message
        $CityResults[$City]['Success'] = $false
        $OverallSuccess = $false
    }
    
    Write-Host ""
}

# Generate rollout summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GENERATING ROLLOUT SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not $DryRun) {
    try {
        $SummaryCommand = @(
            "python", "pipelines/generate_v3_rollout_summary.py",
            "--cities"
        ) + $Cities + @(
            "--year", $Year.ToString(),
            "--config", $Config
        )
        
        if ($Verbose) { $SummaryCommand += "--verbose" }
        
        Write-Host "Generating comprehensive rollout summary..." -ForegroundColor White
        $SummaryProcess = Start-Process -FilePath "python" -ArgumentList ($SummaryCommand[1..($SummaryCommand.Length-1)]) -Wait -PassThru -NoNewWindow
        
        if ($SummaryProcess.ExitCode -eq 0) {
            Write-Host "✅ Rollout summary generated successfully" -ForegroundColor Green
            
            # Find and display the generated files
            $LatestSummary = Get-ChildItem -Path "reports/runs" -Filter "*_v3_rollout_summary.md" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $LatestCSV = Get-ChildItem -Path "reports/runs" -Filter "*_v3_rollout_table.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            
            if ($LatestSummary) {
                Write-Host "📄 Summary Report: $($LatestSummary.FullName)" -ForegroundColor Cyan
            }
            if ($LatestCSV) {
                Write-Host "📊 CSV Table: $($LatestCSV.FullName)" -ForegroundColor Cyan
            }
            
        } else {
            Write-Host "⚠️ Summary generation failed (exit code: $($SummaryProcess.ExitCode))" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "❌ Error generating summary: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "[DRY RUN] Would generate rollout summary for cities: $($Cities -join ', ')" -ForegroundColor Yellow
}

# Final summary
$EndTime = Get-Date
$TotalDuration = ($EndTime - $StartTime).TotalMinutes

$SuccessfulCities = @($CityResults.Values | Where-Object { $_.Success -eq $true })
$FailedCities = @($CityResults.Values | Where-Object { $_.Success -eq $false })

Write-Host ""
Write-Host "========================================" -ForegroundColor White
Write-Host "ROLLOUT COMPLETE" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host "Total Duration: $([math]::Round($TotalDuration, 1)) minutes" -ForegroundColor White
Write-Host "Cities Processed: $($Cities.Count)" -ForegroundColor White
Write-Host "Successful: $($SuccessfulCities.Count)" -ForegroundColor $(if ($SuccessfulCities.Count -eq $Cities.Count) { 'Green' } else { 'Yellow' })
Write-Host "Failed: $($FailedCities.Count)" -ForegroundColor $(if ($FailedCities.Count -gt 0) { 'Red' } else { 'Green' })

# Detailed city results table
Write-Host ""
Write-Host "City Results:" -ForegroundColor White
Write-Host "City          Status      Duration   v3 Extract  Compare    A/B Eval   Issues" -ForegroundColor Gray
Write-Host "----          ------      --------   ----------  -------    --------   ------" -ForegroundColor Gray

foreach ($City in $Cities) {
    $Result = $CityResults[$City]
    $Status = if ($Result.Success) { "SUCCESS " } else { "FAILED  " }
    $StatusColor = if ($Result.Success) { "Green" } else { "Red" }
    
    $Duration = if ($Result.Duration) { "$([math]::Round($Result.Duration, 0))s".PadLeft(8) } else { "N/A".PadLeft(8) }
    
    # Extract step timings
    $EnvTime = if ($Result.Steps['env_v3'] -and $Result.Steps['env_v3'].Duration) { "$([math]::Round($Result.Steps['env_v3'].Duration, 0))s" } else { "N/A" }
    $CompTime = if ($Result.Steps['compare_v2_v3'] -and $Result.Steps['compare_v2_v3'].Duration) { "$([math]::Round($Result.Steps['compare_v2_v3'].Duration, 0))s" } else { "N/A" }
    $ABTime = if ($Result.Steps['ab_eval'] -and $Result.Steps['ab_eval'].Duration) { "$([math]::Round($Result.Steps['ab_eval'].Duration, 0))s" } else { "N/A" }
    
    $EnvTime = $EnvTime.PadLeft(10)
    $CompTime = $CompTime.PadLeft(7)
    $ABTime = $ABTime.PadLeft(8)
    
    # Issues count
    $Issues = 0
    if ($Result.ContainsKey('Error')) { $Issues++ }
    if ($Result.ContainsKey('MissingFiles')) { $Issues++ }
    foreach ($Step in $Result.Steps.Values) {
        if (-not $Step.Success -and -not $Step.Skipped) { $Issues++ }
    }
    
    $CityName = $City.PadRight(12)
    Write-Host "$CityName" -NoNewline
    Write-Host "$Status" -ForegroundColor $StatusColor -NoNewline
    Write-Host "$Duration $EnvTime $CompTime $ABTime   $Issues"
}

Write-Host ""

# Exit with appropriate code
if ($OverallSuccess -and $SuccessfulCities.Count -eq $Cities.Count) {
    Write-Host "🎉 All cities processed successfully!" -ForegroundColor Green
    exit 0
} elseif ($SuccessfulCities.Count -gt 0) {
    Write-Host "⚠️ Partial success - some cities failed" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "❌ All cities failed" -ForegroundColor Red
    exit 2
}