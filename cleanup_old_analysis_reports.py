#!/usr/bin/env python3
"""
Cleanup Old Analysis Reports

This script cleans up old integrated analysis backup files to save memory.
Keeps only the 1 most recent backup.
"""

import os
from pathlib import Path
from datetime import datetime

def cleanup_analysis_reports():
    """Clean up old analysis reports"""
    
    report_dir = Path("tests/reports")
    
    if not report_dir.exists():
        print("❌ No reports directory found")
        return
    
    print("🧹 Cleaning up old analysis reports...")
    
    # Find all backup files
    backup_files = list(report_dir.glob("integrated_analysis_backup_*.json"))
    
    if not backup_files:
        print("✅ No backup files found to clean up")
        return
    
    print(f"📊 Found {len(backup_files)} backup files:")
    for f in backup_files:
        size_kb = f.stat().st_size / 1024
        print(f"   📄 {f.name} ({size_kb:.1f} KB)")
    
    if len(backup_files) <= 1:
        print("✅ Only 1 or fewer backups found, no cleanup needed")
        return
    
    # Sort by filename (timestamp) - newer files have later timestamps
    backup_files.sort(key=lambda x: x.name, reverse=True)
    
    # Keep only the 1 most recent
    files_to_keep = backup_files[:1]
    files_to_delete = backup_files[1:]
    
    print(f"\n🎯 Cleanup plan:")
    print(f"   ✅ Keep {len(files_to_keep)} most recent backup:")
    for f in files_to_keep:
        print(f"      📄 {f.name}")
    
    print(f"   🗑️  Delete {len(files_to_delete)} old backups:")
    for f in files_to_delete:
        size_kb = f.stat().st_size / 1024
        print(f"      📄 {f.name} ({size_kb:.1f} KB)")
    
    # Calculate space savings
    total_deleted_size = sum(f.stat().st_size for f in files_to_delete) / 1024
    
    # Ask for confirmation
    response = input(f"\n💾 This will free up {total_deleted_size:.1f} KB. Continue? (y/N): ")
    
    if response.lower() not in ['y', 'yes']:
        print("❌ Cleanup cancelled")
        return
    
    # Delete old files
    deleted_count = 0
    for old_backup in files_to_delete:
        try:
            old_backup.unlink()
            print(f"🗑️  Deleted: {old_backup.name}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Failed to delete {old_backup.name}: {e}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   🗑️  Deleted {deleted_count} old backup files")
    print(f"   💾 Freed up {total_deleted_size:.1f} KB")
    print(f"   📄 Kept {len(files_to_keep)} most recent backup")
    
    # Show remaining files
    remaining_backups = list(report_dir.glob("integrated_analysis_backup_*.json"))
    if remaining_backups:
        print(f"\n📊 Remaining backup files:")
        for f in remaining_backups:
            size_kb = f.stat().st_size / 1024
            print(f"   📄 {f.name} ({size_kb:.1f} KB)")

def show_current_reports():
    """Show current analysis reports"""
    
    report_dir = Path("tests/reports")
    
    if not report_dir.exists():
        print("❌ No reports directory found")
        return
    
    print("📊 Current Analysis Reports:")
    print("=" * 50)
    
    # Current reports
    current_files = [
        "integrated_analysis_current.json",
        "integrated_analysis_current.html",
        "integrated_analysis.db"
    ]
    
    for filename in current_files:
        file_path = report_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            modified = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ {filename}")
            print(f"   📏 Size: {size_kb:.1f} KB")
            print(f"   📅 Modified: {modified}")
        else:
            print(f"❌ {filename} (not found)")
    
    # Backup files
    backup_files = list(report_dir.glob("integrated_analysis_backup_*.json"))
    if backup_files:
        backup_files.sort(key=lambda x: x.name, reverse=True)
        print(f"\n💾 Backup Files ({len(backup_files)}):")
        total_backup_size = 0
        for f in backup_files:
            size_kb = f.stat().st_size / 1024
            total_backup_size += size_kb
            modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   📄 {f.name} ({size_kb:.1f} KB, {modified})")
        print(f"   📊 Total backup size: {total_backup_size:.1f} KB")
    else:
        print("\n💾 No backup files found")

def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--show', '-s', 'show']:
            show_current_reports()
            return
        elif sys.argv[1] in ['--help', '-h', 'help']:
            print("Usage:")
            print("  python cleanup_old_analysis_reports.py           # Clean up old backups")
            print("  python cleanup_old_analysis_reports.py --show    # Show current reports")
            print("  python cleanup_old_analysis_reports.py --help    # Show this help")
            return
    
    cleanup_analysis_reports()

if __name__ == "__main__":
    main() 