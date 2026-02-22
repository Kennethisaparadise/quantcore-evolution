# QuantCore - Quick Rebuild Guide

## If Things Break

### Prerequisites
```bash
# Install Node.js (v22+) and npm
node -v  # should be v22+

# Install pnpm (recommended) or use npm
npm install -g pnpm
```

### Quick Rebuild
```bash
cd /home/kenner/clawd/quantcore-dashboard

# Install dependencies
pnpm install
# or: npm install

# Start dev server
pnpm dev -- --port 5179
# or: npm run dev -- --port 5179
```

### If Backup Available
```bash
# Restore from backup
cd /home/kenner/clawd
tar -xzf backups/quantcore/quantcore_YYYYMMDD_HHMMSS.tar.gz
```

## Project Structure
```
quantcore-dashboard/
├── src/
│   ├── App.tsx              # Main UI component
│   └── engine/
│       ├── mutations/       # 66 mutation operators
│       ├── fitness/         # Fitness functions
│       ├── selection/       # Selection operators
│       ├── crossover/       # Crossover operators
│       ├── backtest/        # Backtesting engine
│       ├── export/          # Export modules (JSON, PineScript, Python, CSV)
│       ├── types/           # TypeScript definitions
│       └── index.ts         # Engine exports
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## Key Files to Backup
- `src/engine/mutations/index.ts` - Mutation operators (66 total)
- `src/engine/types/index.ts` - Type definitions
- `src/engine/fitness/` - Fitness calculation
- `src/engine/backtest/` - Backtest engine
- `src/App.tsx` - Main UI

## Running
```bash
cd /home/kenner/clawd/quantcore-dashboard
pnpm dev -- --port 5179
# Open http://localhost:5179
```

## Git (for version control)
```bash
cd /home/kenner/clawd/quantcore-dashboard
git add .
git commit -m "Checkpoint: description"
git push  # if remote configured
```
