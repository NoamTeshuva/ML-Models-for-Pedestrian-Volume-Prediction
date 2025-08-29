#!/usr/bin/env python3
"""
Simple script to verify all required files exist before starting the server.
Run this during build to catch missing files early.
"""
import os
import sys

def verify_files():
    """Verify all required files exist"""
    required_files = [
        'app.py',
        'osm_tiles.py',
        'cb_model.cbm',
        'requirements.txt'
    ]
    
    print("=== File Verification ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"✗ {file} - MISSING")
            missing_files.append(file)
    
    print("\n=== Directory Contents ===")
    try:
        files = os.listdir('.')
        for f in sorted(files):
            if f.endswith('.py') or f.endswith('.cbm') or f == 'requirements.txt':
                print(f"  {f}")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files found!")
        return True

if __name__ == "__main__":
    if not verify_files():
        sys.exit(1)