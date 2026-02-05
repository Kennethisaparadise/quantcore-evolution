# MEMORY.md - Curated Long-Term Memory

## Project Context
- Working on MetaForge Pro: A Star Wars Unlimited deck building and analysis application
- Focus on deck optimization, meta analysis, and strategic planning
- Components include deck builder, lab simulation, database, matchups, and tournament tools

## Key Features Implemented
1. Deck Builder with card search and composition tools
2. The Lab with evolutionary simulation and deck optimization
3. Database with card catalog and meta statistics
4. Matchup analysis tools
5. Tournament preparation features
6. Meta forecasting and prediction algorithms
7. Deck health scoring system
8. Export and sharing functionality
9. Notification system for meta shifts

## Technical Approach
- Using React with TypeScript for frontend development
- Implementing with shadcn/ui components for consistent UI
- Supabase integration for backend services
- Responsive design for cross-device compatibility
- Modular component architecture for maintainability
- Canvas-based image generation for deck exports

## User Experience Philosophy
- Focus on Star Wars Unlimited players optimizing their decks
- Emphasis on meta awareness and strategic planning
- Comprehensive analysis tools for deck performance
- Visual feedback and tactical recommendations

## TrustyLaw Hub Project
- Legal technology platform for law firms and legal professionals
- Similar React/TypeScript architecture with shadcn/ui components
- Supabase integration for document management and client data
- Key focus on document handling, legal workflows, and client portals

## Common Debugging Issues Found
- Invalid icon imports from lucide-react library (e.g., FilePdf, Building2, Card as CardComponent)
- These cause build failures and runtime errors when components try to render non-existent icons
- Solution: Replace with valid lucide-react exports (FileIcon, Building, Card component)
- Always verify icon names exist in the lucide-react library before importing

## Deck/Card Import Implementation
- Successfully implemented comprehensive import functionality for MetaForge Pro
- Created SwudbImporter, CardImporter, and DeckImportService for handling various import formats
- Added UI enhancements to DeckBuilder with advanced import modal
- Supports SWUDB URLs, text lists, JSON, and CSV formats
- Includes comprehensive error handling and validation
- Seamlessly integrated with existing deck saving system

## Debugging Methodology
- Systematic first-principles approach to identify root causes
- Verification through successful build processes and development server operation
- Comprehensive documentation of fixes in summary files
- Cross-application pattern recognition for similar issues

## Moltbook Presence
- Registered AI agent "Alient" on Moltbook (AI agent social network)
- Profile: https://www.moltbook.com/u/Alient
- Ready to post SWU meta analysis and legal resources content
- Sharing strategy: Post competitive meta updates and legal tech insights

## SWU Meta Resources
- Primary sources: swumetastats.com, swu-competitivehub.com, starwarsunlimited.gg
- Key meta data points tracked for competitive analysis
- Focus on tier list updates and tournament-winning decklists

## Polymarket Trading Bot
- Built automated trading bot for Polymarket prediction markets
- GitHub repository: polymarket-trading-bot (local repo created)
- Public API integration (no keys needed for data):
  - Gamma API: https://gamma-api.polymarket.com
  - CLOB API: https://clob.polymarket.com
- Features: market discovery, price tracking, arbitrage detection
- Configuration via config.json with Telegram notifications support
- Python-based with modular architecture (api/, bots/, utils/)