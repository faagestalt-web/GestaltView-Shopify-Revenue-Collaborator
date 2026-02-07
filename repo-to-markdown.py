#!/usr/bin/env python3
"""
Repository to Markdown Converter
Converts entire repository into a single markdown file for LLM collaboration
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import mimetypes

# Configuration
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "exports"
OUTPUT_FILE = OUTPUT_DIR / f"repo-snapshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"

# Files and directories to ignore
IGNORE_PATTERNS = {
    "node_modules",
    ".next",
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    ".env.local",
    ".env.production",
    "dist",
    "build",
    "coverage",
    ".cache",
    ".vercel",
    ".snapshots",
    ".vscode",
    ".devcontainer",
    "exports",
    "logs",
    ".sponsor_me",
    ".context",
}

IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".sqlite",
    ".db",
    ".min.js",
    ".min.css",
    ".map",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".tsbuildinfo",
}

IGNORE_FILES = {
    "package-lock.json",
    "tsconfig.tsbuildinfo",
    ".DS_Store",
    ".gitignore",
    ".gitattributes",
}

# Language mappings for syntax highlighting
LANGUAGE_MAP = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".py": "python",
    ".sh": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
    ".sql": "sql",
    ".env": "bash",
    ".txt": "text",
    ".toml": "toml",
    ".ini": "ini",
    ".conf": "nginx",
    ".Dockerfile": "dockerfile",
}

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB


def should_ignore(path: Path) -> bool:
    """Check if file or directory should be ignored."""
    # Check if any parent directory is in ignore patterns
    for part in path.parts:
        if part in IGNORE_PATTERNS:
            return True
    
    # Check file extension
    if path.suffix in IGNORE_EXTENSIONS:
        return True
    
    # Check file name
    if path.name in IGNORE_FILES:
        return True
    
    # Ignore hidden files
    if path.name.startswith('.') and path.name not in {'.env.example', '.python-version'}:
        return True
    
    return False


def get_language(path: Path) -> str:
    """Get language identifier for syntax highlighting."""
    return LANGUAGE_MAP.get(path.suffix, "")


def is_text_file(path: Path) -> bool:
    """Check if file is a text file."""
    try:
        # Try to read first few bytes
        with open(path, 'r', encoding='utf-8') as f:
            f.read(512)
        return True
    except (UnicodeDecodeError, PermissionError):
        return False


def format_size(size: int) -> str:
    """Format size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def generate_tree_structure(root: Path, prefix: str = "", max_depth: int = 4, current_depth: int = 0) -> list:
    """Generate a tree structure of the repository."""
    if current_depth >= max_depth:
        return []
    
    lines = []
    try:
        items = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        items = [item for item in items if not should_ignore(item.relative_to(REPO_ROOT))]
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            display_name = item.name + ("/" if item.is_dir() else "")
            lines.append(f"{prefix}{current_prefix}{display_name}")
            
            if item.is_dir():
                lines.extend(
                    generate_tree_structure(
                        item,
                        prefix + next_prefix,
                        max_depth,
                        current_depth + 1
                    )
                )
    except PermissionError:
        pass
    
    return lines


def main():
    print("🚀 Repository to Markdown Converter")
    print("=" * 40)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize counters
    file_count = 0
    total_size = 0
    skipped_files = []
    
    # Start writing markdown
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as md:
        # Write header
        md.write("# Repository Snapshot\n\n")
        md.write("> Generated for LLM collaboration and documentation purposes\n\n")
        md.write("## Table of Contents\n\n")
        md.write("- [Project Overview](#project-overview)\n")
        md.write("- [Repository Structure](#repository-structure)\n")
        md.write("- [File Contents](#file-contents)\n\n")
        md.write("---\n\n")
        
        # Project overview
        md.write("## Project Overview\n\n")
        md.write("**Repository:** ShopifyRevenueAgent\n")
        md.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Repository structure
        print("📁 Analyzing repository structure...")
        md.write("## Repository Structure\n\n")
        md.write("```\n")
        md.write(".\n")
        tree_lines = generate_tree_structure(REPO_ROOT)
        md.write("\n".join(tree_lines))
        md.write("\n```\n\n")
        md.write("---\n\n")
        
        # File contents
        md.write("## File Contents\n\n")
        
        print("📝 Processing files...\n")
        
        # Walk through all files
        for file_path in sorted(REPO_ROOT.rglob("*")):
            if file_path.is_file():
                # Get relative path
                try:
                    rel_path = file_path.relative_to(REPO_ROOT)
                except ValueError:
                    continue
                
                # Skip if should be ignored
                if should_ignore(rel_path):
                    continue
                
                # Check file size
                file_size = file_path.stat().st_size
                if file_size > MAX_FILE_SIZE:
                    print(f"⚠️  Skipping large file: {rel_path} ({format_size(file_size)})")
                    skipped_files.append((str(rel_path), "too large"))
                    continue
                
                # Check if text file
                if not is_text_file(file_path):
                    skipped_files.append((str(rel_path), "binary"))
                    continue
                
                print(f"✓ Adding: {rel_path}")
                
                # Add file to markdown
                lang = get_language(file_path)
                md.write(f"\n### `{rel_path}`\n\n")
                md.write(f"```{lang}\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        md.write(content)
                        if not content.endswith('\n'):
                            md.write('\n')
                except Exception as e:
                    md.write(f"# Error reading file: {e}\n")
                
                md.write("```\n\n")
                md.write("---\n\n")
                
                file_count += 1
                total_size += file_size
        
        # Add summary
        md.write("## Summary\n\n")
        md.write(f"- **Total Files Processed:** {file_count}\n")
        md.write(f"- **Total Size:** {format_size(total_size)}\n")
        md.write(f"- **Files Skipped:** {len(skipped_files)}\n")
        md.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if skipped_files:
            md.write("### Skipped Files\n\n")
            for file, reason in skipped_files[:20]:  # Show first 20
                md.write(f"- `{file}` ({reason})\n")
            if len(skipped_files) > 20:
                md.write(f"\n... and {len(skipped_files) - 20} more\n")
            md.write("\n")
        
        md.write("---\n\n")
        md.write("*This snapshot was generated for LLM collaboration. ")
        md.write("Some files may be excluded based on ignore patterns.*\n")
    
    # Print summary
    print()
    print("✅ Conversion complete!")
    print("📊 Statistics:")
    print(f"   • Files processed: {file_count}")
    print(f"   • Total size: {format_size(total_size)}")
    print(f"   • Files skipped: {len(skipped_files)}")
    print(f"   • Output file: {OUTPUT_FILE}")
    print()
    print("💡 You can now use this file for:")
    print("   • LLM context (ChatGPT, Claude, etc.)")
    print("   • Code reviews")
    print("   • Documentation")
    print("   • Onboarding new developers")
    print()


if __name__ == "__main__":
    main()
