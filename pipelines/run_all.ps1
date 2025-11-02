# run_all.ps1 - Master Pipeline Runner
# Complete pedestrian volume prediction pipeline from feature extraction to inference

param(
    [Parameter(Mandatory=$true)]
    [string]$City,
    
    [Parameter(Mandatory=$false)]
    [int]$Year = 2023,
    
    [Parameter(Mandatory=$false)]
    [string]$Mode = "full",  # full, training, inference, features_only
    
    [Parameter(Mandatory=$false)]
    [string]$Features = "baseline+environmental",
    
    [Parameter(Mandatory=$false)]
    [string]$CvStrategy = "leave_one_city_out",
    
    [Parameter(Mandatory=$false)]
    [string]$Cities,  # Comma-separated list for multi-city training
    
    [Parameter(Mandatory=$false)]
    [string]$InferenceStart,  # Start date for inference
    
    [Parameter(Mandatory=$false)]
    [string]$InferenceEnd,    # End date for inference
    
    [Parameter(Mandatory=$false)]
    [string]$ConfigPath = "config/cities.yaml",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipFeatureExtraction,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTraining,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Set error handling
$ErrorActionPreference = "Stop"

# Function to write timestamped log messages
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $(
        switch ($Level) {
            "ERROR" { "Red" }
            "WARN"  { "Yellow" }
            "SUCCESS" { "Green" }
            default { "White" }
        }
    )
}

# Function to check if Python command exists
function Test-PythonCommand {
    try {
        python --version | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to run Python command with error handling
function Invoke-PythonCommand {
    param([string]$Command, [string]$Description)
    
    Write-Log "Starting: $Description" "INFO"
    Write-Log "Command: $Command" "INFO"
    
    $startTime = Get-Date
    
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        Write-Log "Completed: $Description (${duration}s)" "SUCCESS"
        return $true
        
    } catch {
        Write-Log "Failed: $Description - $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Function to check configuration and paths
function Test-Configuration {
    param([string]$City, [int]$Year)
    
    Write-Log "Validating configuration for $City $Year" "INFO"
    
    # Check config file exists
    if (-not (Test-Path $ConfigPath)) {
        Write-Log "Configuration file not found: $ConfigPath" "ERROR"
        return $false
    }
    
    # Try to load and parse config (basic validation)
    try {
        python -c "import yaml; yaml.safe_load(open('$ConfigPath'))" 2>$null
    } catch {
        Write-Log "Invalid YAML configuration: $ConfigPath" "ERROR"
        return $false
    }
    
    Write-Log "Configuration validation passed" "SUCCESS"
    return $true
}

# Function to extract environmental features
function Start-FeatureExtraction {
    param([string]$City, [int]$Year)
    
    Write-Log "=== FEATURE EXTRACTION PHASE ===" "INFO"
    
    # Run environmental feature extraction using v2 scripts
    $success = Invoke-PythonCommand -Command ".\scripts\recompute_env_features.ps1 -City $City -Year $Year" -Description "Environmental feature extraction"
    
    if (-not $success) {
        Write-Log "Feature extraction failed for $City" "ERROR"
        return $false
    }
    
    # Verify outputs exist
    $outputDir = "data\processed\$City\csv"
    $requiredFiles = @(
        "$outputDir\topography_${Year}_${City}.csv",
        "$outputDir\green_canopy_${Year}_${City}.csv"
    )
    
    $missingFiles = @()
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            $missingFiles += $file
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Log "Missing output files: $($missingFiles -join ', ')" "WARN"
    } else {
        Write-Log "All feature files generated successfully" "SUCCESS"
    }
    
    return $true
}

# Function to run training pipeline
function Start-Training {
    param([string]$TrainingCities, [string]$Features, [string]$CvStrategy)
    
    Write-Log "=== TRAINING PHASE ===" "INFO"
    
    # Parse cities list
    $citiesList = $TrainingCities -split ","
    $citiesArg = $citiesList -join " "
    
    Write-Log "Training cities: $citiesArg" "INFO"
    Write-Log "Features: $Features" "INFO"
    Write-Log "CV Strategy: $CvStrategy" "INFO"
    
    # Run training pipeline
    $command = "python pipelines\run_training.py --cities $citiesArg --features $Features --cv-strategy $CvStrategy --config $ConfigPath"
    
    $success = Invoke-PythonCommand -Command $command -Description "Model training"
    
    if (-not $success) {
        Write-Log "Training failed" "ERROR"
        return $false
    }
    
    # Find the most recent model files
    $modelFiles = Get-ChildItem -Path "data\models" -Filter "*.cbm" | Sort-Object LastWriteTime -Descending
    
    if ($modelFiles.Count -eq 0) {
        Write-Log "No model files found after training" "WARN"
    } else {
        Write-Log "Training completed. Model files:" "SUCCESS"
        foreach ($modelFile in $modelFiles | Select-Object -First 3) {
            Write-Log "  $($modelFile.Name)" "INFO"
        }
    }
    
    return $true
}

# Function to run inference
function Start-Inference {
    param([string]$City, [string]$StartDate, [string]$EndDate)
    
    Write-Log "=== INFERENCE PHASE ===" "INFO"
    
    # Find the most recent model
    $modelFiles = Get-ChildItem -Path "data\models" -Filter "*.cbm" | Sort-Object LastWriteTime -Descending
    
    if ($modelFiles.Count -eq 0) {
        Write-Log "No model files found for inference" "ERROR"
        return $false
    }
    
    $modelPath = $modelFiles[0].FullName
    Write-Log "Using model: $($modelFiles[0].Name)" "INFO"
    
    # Set default dates if not provided
    if (-not $StartDate) {
        $StartDate = (Get-Date).ToString("yyyy-MM-dd")
    }
    if (-not $EndDate) {
        $EndDate = (Get-Date).AddDays(7).ToString("yyyy-MM-dd")
    }
    
    Write-Log "Inference period: $StartDate to $EndDate" "INFO"
    
    # Run inference
    $command = "python pipelines\run_inference.py --city $City --model `"$modelPath`" --start-date $StartDate --end-date $EndDate --config $ConfigPath"
    
    $success = Invoke-PythonCommand -Command $command -Description "Inference"
    
    if (-not $success) {
        Write-Log "Inference failed" "ERROR"
        return $false
    }
    
    # Check for prediction outputs
    $predictionFiles = Get-ChildItem -Path "reports\predictions" -Filter "*.csv" | Sort-Object LastWriteTime -Descending
    
    if ($predictionFiles.Count -gt 0) {
        Write-Log "Predictions saved: $($predictionFiles[0].Name)" "SUCCESS"
    }
    
    return $true
}

# Function to run quality assurance checks
function Start-QualityAssurance {
    param([string]$City, [int]$Year)
    
    Write-Log "=== QUALITY ASSURANCE ===" "INFO"
    
    # Run the debug script to check coverage
    $success = Invoke-PythonCommand -Command "python scripts\debug_env_nulls.py --city $City --year $Year" -Description "Quality assurance checks"
    
    if (-not $success) {
        Write-Log "Quality assurance failed" "WARN"
        return $false
    }
    
    return $true
}

# Function to generate run summary
function New-RunSummary {
    param([hashtable]$Results, [string]$City, [int]$Year, [string]$RunId)
    
    $summaryPath = "reports\runs\${RunId}_pipeline_summary.md"
    
    $summary = @"
# Pipeline Run Summary

**Run ID**: $RunId
**City**: $City
**Year**: $Year
**Timestamp**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Mode**: $Mode

## Execution Summary

"@

    foreach ($phase in $Results.Keys) {
        $status = if ($Results[$phase]) { "✅ SUCCESS" } else { "❌ FAILED" }
        $summary += "- **$phase**: $status`n"
    }
    
    $summary += @"

## Output Files

### Models
"@
    
    $modelFiles = Get-ChildItem -Path "data\models" -Filter "*${RunId}*.cbm" -ErrorAction SilentlyContinue
    if ($modelFiles) {
        foreach ($file in $modelFiles) {
            $summary += "- $($file.Name)`n"
        }
    } else {
        $summary += "- None generated`n"
    }
    
    $summary += @"

### Reports
"@
    
    $reportFiles = @(
        Get-ChildItem -Path "reports\metrics" -Filter "*${RunId}*" -ErrorAction SilentlyContinue
        Get-ChildItem -Path "reports\feature_importance" -Filter "*${RunId}*" -ErrorAction SilentlyContinue
        Get-ChildItem -Path "reports\predictions" -Filter "*${RunId}*" -ErrorAction SilentlyContinue
    )
    
    if ($reportFiles) {
        foreach ($file in $reportFiles) {
            $summary += "- $($file.Name)`n"
        }
    } else {
        $summary += "- None generated`n"
    }
    
    $summary += @"

## Next Steps

"@
    
    if ($Results["Training"] -and $Results["FeatureExtraction"]) {
        $summary += "- Models are ready for inference on new data`n"
        $summary += "- Run inference with: ``.\pipelines\run_inference.py --city $City --model <model_path>```n"
    }
    
    if (-not $Results["FeatureExtraction"]) {
        $summary += "- Fix feature extraction issues before training`n"
    }
    
    if (-not $Results["Training"]) {
        $summary += "- Review training data quality and model parameters`n"
    }
    
    $summary += "- Review detailed logs and error messages above`n"
    
    # Save summary
    New-Item -ItemType Directory -Path "reports\runs" -Force | Out-Null
    $summary | Out-File -FilePath $summaryPath -Encoding UTF8
    
    Write-Log "Run summary saved: $summaryPath" "INFO"
    
    return $summaryPath
}

# Main execution
function Main {
    $runId = Get-Date -Format "yyyyMMdd_HHmmss"
    $results = @{}
    
    Write-Log "=== PEDESTRIAN VOLUME PREDICTION PIPELINE ===" "INFO"
    Write-Log "Run ID: $runId" "INFO"
    Write-Log "City: $City" "INFO"
    Write-Log "Year: $Year" "INFO"
    Write-Log "Mode: $Mode" "INFO"
    Write-Log "Features: $Features" "INFO"
    
    # Pre-flight checks
    if (-not (Test-PythonCommand)) {
        Write-Log "Python not found in PATH" "ERROR"
        exit 1
    }
    
    if (-not (Test-Configuration -City $City -Year $Year)) {
        Write-Log "Configuration validation failed" "ERROR"
        exit 1
    }
    
    # Parse training cities
    $trainingCities = if ($Cities) { $Cities } else { $City }
    
    try {
        # Phase 1: Feature Extraction
        if ($Mode -in @("full", "features_only") -and -not $SkipFeatureExtraction) {
            $results["FeatureExtraction"] = Start-FeatureExtraction -City $City -Year $Year
        } else {
            Write-Log "Skipping feature extraction" "INFO"
            $results["FeatureExtraction"] = $true
        }
        
        # Phase 2: Quality Assurance
        if ($Mode -in @("full", "features_only") -and $results["FeatureExtraction"]) {
            $results["QualityAssurance"] = Start-QualityAssurance -City $City -Year $Year
        }
        
        # Phase 3: Training
        if ($Mode -in @("full", "training") -and -not $SkipTraining) {
            if ($results["FeatureExtraction"] -or $SkipFeatureExtraction) {
                $results["Training"] = Start-Training -TrainingCities $trainingCities -Features $Features -CvStrategy $CvStrategy
            } else {
                Write-Log "Skipping training due to feature extraction failure" "WARN"
                $results["Training"] = $false
            }
        } else {
            Write-Log "Skipping training" "INFO"
            $results["Training"] = $true
        }
        
        # Phase 4: Inference (optional)
        if ($Mode -in @("full", "inference") -and $InferenceStart) {
            if ($results["Training"] -or $SkipTraining) {
                $results["Inference"] = Start-Inference -City $City -StartDate $InferenceStart -EndDate $InferenceEnd
            } else {
                Write-Log "Skipping inference due to training failure" "WARN"
                $results["Inference"] = $false
            }
        }
        
    } catch {
        Write-Log "Pipeline execution failed: $($_.Exception.Message)" "ERROR"
        exit 1
    }
    
    # Generate summary
    $summaryPath = New-RunSummary -Results $results -City $City -Year $Year -RunId $runId
    
    # Final status
    $allSuccessful = $true
    foreach ($phase in $results.Keys) {
        if (-not $results[$phase]) {
            $allSuccessful = $false
            break
        }
    }
    
    if ($allSuccessful) {
        Write-Log "=== PIPELINE COMPLETED SUCCESSFULLY ===" "SUCCESS"
        Write-Log "Summary: $summaryPath" "INFO"
        exit 0
    } else {
        Write-Log "=== PIPELINE COMPLETED WITH ERRORS ===" "ERROR"
        Write-Log "Summary: $summaryPath" "INFO"
        exit 1
    }
}

# Usage examples in comments
<#
# Basic usage - extract features and train for single city
.\pipelines\run_all.ps1 -City melbourne -Year 2019

# Multi-city training
.\pipelines\run_all.ps1 -City melbourne -Cities "melbourne,new_york,zurich" -Features "baseline+environmental+network"

# Full pipeline with inference
.\pipelines\run_all.ps1 -City melbourne -Mode full -InferenceStart "2023-12-01" -InferenceEnd "2023-12-07"

# Features only
.\pipelines\run_all.ps1 -City melbourne -Mode features_only -Year 2023

# Training only (skip feature extraction)
.\pipelines\run_all.ps1 -City melbourne -Mode training -SkipFeatureExtraction -Cities "melbourne,new_york"

# Inference only (use existing model)
.\pipelines\run_all.ps1 -City melbourne -Mode inference -InferenceStart "2023-12-01" -InferenceEnd "2023-12-31"
#>

# Execute main function
Main