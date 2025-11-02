# PowerShell orchestrator for Fast Canopy processing pipeline
# Runs all 3 steps to promote FAST probe -> edge -> sensor -> base table

param(
    [Parameter(Mandatory=$true)]
    [string]$City,
    
    [Parameter(Mandatory=$true)]
    [string]$Year
)

Write-Host "=== Fast Canopy Processing Pipeline ===" -ForegroundColor Green
Write-Host "City: $City"
Write-Host "Year: $Year"
Write-Host ""

# Define paths
$cityLower = $City.ToLower()
$fastProbeInput = "data\processed\$cityLower\csv\green_canopy_FAST_${Year}_${cityLower}_clip.csv"
$edgeOutput = "data\processed\$cityLower\csv\green_canopy_${Year}_${cityLower}_FAST_edge.csv"
$sensorOutput = "data\processed\$cityLower\csv\sensor_env_features_${Year}_${cityLower}_FAST.csv"

Write-Host "Input FAST probe: $fastProbeInput"
Write-Host "Edge output: $edgeOutput" 
Write-Host "Sensor output: $sensorOutput"
Write-Host ""

# Check if input exists
if (!(Test-Path $fastProbeInput)) {
    Write-Host "ERROR: FAST probe input file not found: $fastProbeInput" -ForegroundColor Red
    Write-Host "Please ensure the FAST probe CSV exists before running this pipeline." -ForegroundColor Red
    exit 1
}

# Step 1: Promote edges
Write-Host "--- Step 1: Promoting FAST probe to edge-level features ---" -ForegroundColor Cyan
Write-Host "Running: python scripts\fast_canopy_promote_edges.py"
Write-Host ""

& python scripts\fast_canopy_promote_edges.py `
    --in $fastProbeInput `
    --out $edgeOutput `
    --ndvi-threshold 0.20

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Step 1 (edge promotion) failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""

# Step 2: Join to sensors
Write-Host "--- Step 2: Joining edges to sensor level ---" -ForegroundColor Cyan
Write-Host "Running: python scripts\fast_canopy_sensor_join.py"
Write-Host ""

& python scripts\fast_canopy_sensor_join.py `
    --edges $edgeOutput `
    --out $sensorOutput

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Step 2 (sensor join) failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""

# Step 3: Merge to base table
Write-Host "--- Step 3: Merging to base feature table ---" -ForegroundColor Cyan
Write-Host "Running: python scripts\fast_canopy_merge_to_base.py"
Write-Host ""

& python scripts\fast_canopy_merge_to_base.py `
    --sens $sensorOutput

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Step 3 (base table merge) failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== Fast Canopy Processing Complete ===" -ForegroundColor Green
Write-Host "All steps completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Generated files:"
Write-Host "- Edge features: $edgeOutput"
Write-Host "- Sensor features: $sensorOutput" 
Write-Host "- Base table with canopy: (auto-generated with _with_canopyFAST suffix)"
Write-Host ""

# List final outputs
Write-Host "Verifying outputs exist:"
if (Test-Path $edgeOutput) {
    $edgeSize = (Get-Item $edgeOutput).Length
    $edgeSizeKB = [math]::Round($edgeSize/1KB, 1)
    Write-Host "  [OK] Edge features: $edgeOutput ($edgeSizeKB KB)" -ForegroundColor Green
} else {
    Write-Host "  [X] Edge features: Missing!" -ForegroundColor Red
}

if (Test-Path $sensorOutput) {
    $sensorSize = (Get-Item $sensorOutput).Length  
    $sensorSizeKB = [math]::Round($sensorSize/1KB, 1)
    Write-Host "  [OK] Sensor features: $sensorOutput ($sensorSizeKB KB)" -ForegroundColor Green
} else {
    Write-Host "  [X] Sensor features: Missing!" -ForegroundColor Red
}

# Look for base table with canopy suffix
$baseTablePattern = "data\processed\$cityLower\csv\*${Year}*${cityLower}*_with_canopyFAST.csv"
$baseTableFiles = Get-ChildItem -Path $baseTablePattern -ErrorAction SilentlyContinue

if ($baseTableFiles) {
    foreach ($file in $baseTableFiles) {
        $baseSize = $file.Length
        $baseSizeKB = [math]::Round($baseSize/1KB, 1)
        Write-Host "  [OK] Base table with canopy: $($file.FullName) ($baseSizeKB KB)" -ForegroundColor Green
    }
} else {
    Write-Host "  [?] Base table with canopy: Not found (pattern: $baseTablePattern)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Fast canopy processing pipeline completed successfully!" -ForegroundColor Green