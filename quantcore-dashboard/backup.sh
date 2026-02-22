#!/bin/bash
# QuantCore Backup Script
# Run: ./backup.sh

BACKUP_DIR="/home/kenner/clawd/backups/quantcore"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/kenner/clawd/quantcore-dashboard"

mkdir -p "$BACKUP_DIR"

# Create archive
echo "Creating backup..."
tar -czf "$BACKUP_DIR/quantcore_${TIMESTAMP}.tar.gz" \
    -C /home/kenner/clawd \
    quantcore-dashboard/src/engine \
    quantcore-dashboard/package.json \
    quantcore-dashboard/tsconfig.json \
    quantcore-dashboard/vite.config.ts 2>/dev/null

# Keep only last 10 backups
ls -t "$BACKUP_DIR"/quantcore_*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm 2>/dev/null

echo "Backup saved: quantcore_${TIMESTAMP}.tar.gz"
echo "Backups kept: $(ls -1 $BACKUP_DIR/*.tar.gz 2>/dev/null | wc -l)"
