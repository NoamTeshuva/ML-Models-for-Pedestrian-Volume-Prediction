#!/usr/bin/env python3
"""
Timing Instrumentation Wrapper

Times execution of any command and saves results to JSON for rollout tracking.
Used to measure processing durations for each stage of the v3 pipeline.

Usage:
    python pipelines/track_time.py --stage env_v3 --out-json timing.json -- python script.py args
    python pipelines/track_time.py --stage sensor_merge --meta '{"city":"melbourne"}' --out-json timing.json -- powershell script.ps1
"""

import time
import json
import os
import argparse
import subprocess
import shlex
from datetime import datetime


def main():
    """Main timing wrapper function."""
    parser = argparse.ArgumentParser(
        description='Time command execution and save results to JSON'
    )
    
    parser.add_argument("--stage", required=True, help="Processing stage name")
    parser.add_argument("--meta", default="{}", help="Additional metadata as JSON string")
    parser.add_argument("--out-json", required=True, help="Output JSON file path")
    parser.add_argument("--cmd", nargs=argparse.REMAINDER, required=True, 
                       help="Command to execute (after --)")
    
    args = parser.parse_args()
    
    # Parse metadata
    try:
        meta = json.loads(args.meta)
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}")
        exit(1)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    
    # Record start time
    start_time = time.time()
    start_timestamp = datetime.utcnow().isoformat() + "Z"
    
    print(f"[timing] Starting {args.stage}: {' '.join(args.cmd)}")
    
    # Execute command
    try:
        # Use subprocess.call to preserve exit codes and handle different shells properly
        exit_code = subprocess.call(args.cmd)
    except Exception as e:
        print(f"[timing] Command execution failed: {e}")
        exit_code = 1
    
    # Record end time  
    end_time = time.time()
    end_timestamp = datetime.utcnow().isoformat() + "Z"
    duration = round(end_time - start_time, 2)
    
    # Prepare timing record
    timing_record = {
        "stage": args.stage,
        "command": args.cmd,
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "duration_seconds": duration,
        "return_code": exit_code,
        "success": exit_code == 0,
        **meta  # Include any additional metadata
    }
    
    # Save timing record
    try:
        with open(args.out_json, "w") as f:
            json.dump(timing_record, f, indent=2)
    except Exception as e:
        print(f"[timing] Failed to save timing record: {e}")
        # Don't fail the entire operation just because timing couldn't be saved
    
    # Log summary
    status = "SUCCESS" if exit_code == 0 else "FAILED"
    print(f"[timing] {args.stage}: {duration}s - {status} (rc={exit_code})")
    
    # Exit with the same code as the wrapped command
    exit(exit_code)


if __name__ == "__main__":
    main()