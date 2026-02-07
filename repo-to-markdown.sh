#!/bin/bash
# ============================================================================
# Repository to Markdown Converter
# Converts entire repository into a single markdown file for LLM collaboration
# ============================================================================

# Configuration
OUTPUT_DIR="exports"
OUTPUT_FILE="${OUTPUT_DIR}/repo-snapshot-$(date +%Y%m%d-%H%M%S).md"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Files and directories to ignore (add more as needed)
IGNORE_PATTERNS=(
  "node_modules"
  ".next"
  ".git"
  ".pytest_cache"
  "__pycache__"
  ".venv"
  "venv"
  ".env"
  ".env.local"
  ".env.production"
  "dist"
  "build"
  "coverage"
  ".cache"
  ".vercel"
  "*.pyc"
  "*.pyo"
  "*.pyd"
  ".DS_Store"
  "*.log"
  "logs"
  "*.sqlite"
  "*.db"
  ".snapshots"
  ".vscode"
  ".devcontainer"
  "exports"
  "*.min.js"
  "*.min.css"
  "*.map"
  "package-lock.json"
  "tsconfig.tsbuildinfo"
  ".gitignore"
  "*.jpg"
  "*.jpeg"
  "*.png"
  "*.gif"
  "*.ico"
  "*.svg"
  "*.woff"
  "*.woff2"
  "*.ttf"
  "*.eot"
)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to check if file should be ignored
should_ignore() {
  local file="$1"
  
  for pattern in "${IGNORE_PATTERNS[@]}"; do
    if [[ "$file" == *"$pattern"* ]]; then
      return 0  # true, should ignore
    fi
  done
  
  return 1  # false, should not ignore
}

# Function to get file extension for syntax highlighting
get_language() {
  local file="$1"
  local ext="${file##*.}"
  
  case "$ext" in
    js) echo "javascript" ;;
    jsx) echo "javascript" ;;
    ts) echo "typescript" ;;
    tsx) echo "typescript" ;;
    py) echo "python" ;;
    sh) echo "bash" ;;
    yml|yaml) echo "yaml" ;;
    json) echo "json" ;;
    md) echo "markdown" ;;
    css) echo "css" ;;
    scss) echo "scss" ;;
    html) echo "html" ;;
    sql) echo "sql" ;;
    env) echo "bash" ;;
    txt) echo "text" ;;
    *) echo "" ;;
  esac
}

# Function to check if file is text-based
is_text_file() {
  local file="$1"
  file -b --mime-type "$file" | grep -q '^text/'
}

echo -e "${BLUE}🚀 Repository to Markdown Converter${NC}"
echo -e "${BLUE}====================================${NC}\n"

# Start writing to output file
cat > "$OUTPUT_FILE" << 'EOF'
# Repository Snapshot

> Generated for LLM collaboration and documentation purposes

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [File Contents](#file-contents)

---

## Project Overview

**Repository:** Resume Rockstar
**Generated:** 
EOF

echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add repository structure
echo -e "${YELLOW}📁 Analyzing repository structure...${NC}"

cat >> "$OUTPUT_FILE" << 'EOF'

## Repository Structure

```
EOF

# Generate tree structure (excluding ignored patterns)
cd "$REPO_ROOT"
tree_ignore_args=""
for pattern in "${IGNORE_PATTERNS[@]}"; do
  tree_ignore_args="$tree_ignore_args -I '$pattern'"
done

# Use tree if available, otherwise use find
if command -v tree &> /dev/null; then
  eval "tree -L 4 $tree_ignore_args" >> "$OUTPUT_FILE"
else
  find . -type d \( -name node_modules -o -name .git -o -name .next -o -name __pycache__ \) -prune -o -print | head -n 200 >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

## File Contents

EOF

# Counter for progress
file_count=0
total_size=0

echo -e "${YELLOW}📝 Processing files...${NC}\n"

# Find all text files and add them to markdown
while IFS= read -r -d '' file; do
  # Skip if should be ignored
  if should_ignore "$file"; then
    continue
  fi
  
  # Get relative path
  rel_path="${file#$REPO_ROOT/}"
  
  # Skip if starts with . (hidden files)
  if [[ "$rel_path" == .* ]]; then
    continue
  fi
  
  # Check if it's a text file
  if is_text_file "$file"; then
    # Get language for syntax highlighting
    lang=$(get_language "$file")
    
    # Get file size
    file_size=$(wc -c < "$file")
    total_size=$((total_size + file_size))
    
    # Skip very large files (>1MB)
    if [ "$file_size" -gt 1048576 ]; then
      echo -e "${YELLOW}⚠️  Skipping large file: $rel_path ($(numfmt --to=iec-i --suffix=B $file_size))${NC}"
      continue
    fi
    
    echo -e "${GREEN}✓${NC} Adding: $rel_path"
    
    # Add file to markdown
    cat >> "$OUTPUT_FILE" << EOF

### \`$rel_path\`

\`\`\`$lang
EOF
    cat "$file" >> "$OUTPUT_FILE"
    cat >> "$OUTPUT_FILE" << 'EOF'
```

---

EOF
    
    file_count=$((file_count + 1))
  fi
done < <(find "$REPO_ROOT" -type f -print0 | sort -z)

# Add footer
cat >> "$OUTPUT_FILE" << EOF

---

## Summary

- **Total Files Processed:** $file_count
- **Total Size:** $(numfmt --to=iec-i --suffix=B $total_size)
- **Generated:** $(date '+%Y-%m-%d %H:%M:%S')

---

*This snapshot was generated for LLM collaboration. Some files may be excluded based on ignore patterns.*
EOF

echo ""
echo -e "${GREEN}✅ Conversion complete!${NC}"
echo -e "${BLUE}📊 Statistics:${NC}"
echo -e "   • Files processed: ${GREEN}$file_count${NC}"
echo -e "   • Total size: ${GREEN}$(numfmt --to=iec-i --suffix=B $total_size)${NC}"
echo -e "   • Output file: ${YELLOW}$OUTPUT_FILE${NC}"
echo ""
echo -e "${BLUE}💡 You can now use this file for:${NC}"
echo -e "   • LLM context (ChatGPT, Claude, etc.)"
echo -e "   • Code reviews"
echo -e "   • Documentation"
echo -e "   • Onboarding new developers"
echo ""
