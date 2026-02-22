import { useState, useCallback, useRef } from 'react';
import { Activity, Play, Pause, Square, RotateCcw, Download, Code, FileJson, Trophy, Dna, GitBranch, BarChart3, Target, Wallet, Layers, Settings, Search, X, ChevronRight, TrendingUp, TrendingDown, Award, Zap } from 'lucide-react';
import { EvolutionEngine, generateMockData, exportToJSON, exportToPineScript, exportToPython, exportTradesToCSV, generateStatisticsSummary, ALL_MUTATIONS, DEFAULT_EVOLUTION_CONFIG, createDefaultStrategyParams } from './engine';

type Tab = 'overview' | 'strategies' | 'evolution' | 'population' | 'mutations' | 'backtest' | 'export';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [symbol, setSymbol] = useState('BTC');
  const [engine, setEngine] = useState<EvolutionEngine | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generation, setGeneration] = useState(0);
  const [bestMutant, setBestMutant] = useState<any>(null);
  const [population, setPopulation] = useState<any[]>([]);
  const [hallOfFame, setHallOfFame] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [selectedMutant, setSelectedMutant] = useState<any>(null);
  const [config, setConfig] = useState({ ...DEFAULT_EVOLUTION_CONFIG });
  const [showExport, setShowExport] = useState(false);
  const [exportFormat, setExportFormat] = useState<'json' | 'pine' | 'python' | 'csv'>('json');
  const engineRef = useRef<EvolutionEngine | null>(null);

  const startEvolution = useCallback(async () => {
    const eng = new EvolutionEngine(config, {
      onGeneration: (state) => {
        setProgress(state.progress);
        setGeneration(state.generation);
        setPopulation([...state.population].slice(0, 20));
        if (state.best_mutant) setBestMutant({ ...state.best_mutant });
        setHistory([...state.history]);
      },
      onNewBest: (mutant) => {
        setBestMutant({ ...mutant });
        setSelectedMutant({ ...mutant });
      },
      onComplete: (state) => {
        setIsRunning(false);
        setHallOfFame(state.hall_of_fame);
        setPopulation([...state.population]);
      },
      onError: (err) => {
        console.error(err);
        setIsRunning(false);
      },
    });
    engineRef.current = eng;
    setEngine(eng);
    setIsRunning(true);
    setIsPaused(false);
    eng.initializePopulation();
    eng.evolve(symbol);
  }, [config, symbol]);

  const pauseEvolution = useCallback(() => {
    engineRef.current?.pause();
    setIsPaused(true);
  }, []);

  const resumeEvolution = useCallback(() => {
    engineRef.current?.resume();
    setIsPaused(false);
  }, []);

  const stopEvolution = useCallback(() => {
    engineRef.current?.stop();
    setIsRunning(false);
    setIsPaused(false);
  }, []);

  const resetEvolution = useCallback(() => {
    engineRef.current?.reset();
    setEngine(null);
    setIsRunning(false);
    setIsPaused(false);
    setProgress(0);
    setGeneration(0);
    setBestMutant(null);
    setPopulation([]);
    setHallOfFame([]);
    setHistory([]);
    setSelectedMutant(null);
  }, []);

  const exportCurrent = useCallback(() => {
    if (!selectedMutant || !selectedMutant.backtest_result) return;
    let content = '';
    let filename = `strategy_${selectedMutant.id.slice(0, 8)}`;
    let mimeType = 'text/plain';
    
    switch (exportFormat) {
      case 'json':
        content = exportToJSON(selectedMutant, selectedMutant.backtest_result);
        filename += '.json';
        mimeType = 'application/json';
        break;
      case 'pine':
        content = exportToPineScript(selectedMutant);
        filename += '.pine';
        break;
      case 'python':
        content = exportToPython(selectedMutant, selectedMutant.backtest_result);
        filename += '.py';
        mimeType = 'text/x-python';
        break;
      case 'csv':
        content = exportTradesToCSV(selectedMutant.backtest_result.trades);
        filename += '_trades.csv';
        mimeType = 'text/csv';
        break;
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedMutant, exportFormat]);

  const formatNumber = (num: number, decimals: number = 2) => num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

  return (
    <div style={{ minHeight: '100vh', background: '#0d1117', color: '#e6edf3', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Header - Compact */}
      <header style={{ background: '#161b22', borderBottom: '1px solid #30363d', padding: '8px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'sticky', top: 0, zIndex: 100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ background: 'linear-gradient(135deg, #f7931a, #00d4ff)', padding: '6px', borderRadius: '6px' }}>
            <Activity size={18} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '16px', fontWeight: 700, margin: 0 }}>QuantCore</h1>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <select value={symbol} onChange={e => setSymbol(e.target.value)} style={{ background: '#21262d', border: '1px solid #30363d', borderRadius: '6px', padding: '6px 12px', color: '#e6edf3', fontSize: '13px' }}>
            {['BTC', 'ETH', 'SOL', 'AVAX', 'ARB', 'OP'].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <span style={{ fontSize: '13px', color: isRunning ? '#3fb950' : '#8b949e', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: isRunning ? '#3fb950' : '#8b949e', display: 'inline-block' }}></span>
            {isRunning ? (isPaused ? 'Paused' : 'Running') : 'Ready'}
          </span>
        </div>
      </header>

      <div style={{ display: 'flex' }}>
        {/* Sidebar - Compact */}
        <aside style={{ width: '140px', background: '#161b22', borderRight: '1px solid #30363d', minHeight: 'calc(100vh - 44px)', position: 'sticky', top: 44 }}>
          <nav style={{ padding: '8px 0' }}>
            {[
              { id: 'overview', icon: Activity, label: 'Overview' },
              { id: 'population', icon: Layers, label: 'Pop' },
              { id: 'mutations', icon: Dna, label: 'Mutations' },
              { id: 'export', icon: Download, label: 'Export' },
              { id: 'strategies', icon: Target, label: 'Strategies' },
              { id: 'evolution', icon: GitBranch, label: 'Evol' },
              { id: 'backtest', icon: BarChart3, label: 'Backtest' },
            ].map(item => (
              <button key={item.id} onClick={() => setActiveTab(item.id as Tab)} style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%', padding: '8px 12px', background: activeTab === item.id ? '#21262d' : 'transparent', border: 'none', color: activeTab === item.id ? '#e6edf3' : '#8b949e', cursor: 'pointer', fontSize: '12px', textAlign: 'left', borderLeft: activeTab === item.id ? '3px solid #f7931a' : '3px solid transparent' }}>
                <item.icon size={14} />{item.label}
              </button>
            ))}
          </nav>
        </aside>

        {/* Main Content - Compact */}
        <main style={{ flex: 1, padding: '16px', overflow: 'auto' }}>
          
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div>
              <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Dashboard</h2>
              
              {/* Stats Cards - Compact Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}>Best Return</div>
                  <div style={{ fontSize: '20px', fontWeight: 600, color: bestMutant?.total_return_pct >= 0 ? '#3fb950' : '#f85149' }}>
                    {bestMutant ? `${formatNumber(bestMutant.total_return_pct)}%` : '--'}
                  </div>
                </div>
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}>Best Sharpe</div>
                  <div style={{ fontSize: '20px', fontWeight: 600 }}>{bestMutant ? formatNumber(bestMutant.sharpe_ratio) : '--'}</div>
                </div>
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}>Generation</div>
                  <div style={{ fontSize: '20px', fontWeight: 600 }}>{generation} / {config.generations}</div>
                </div>
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ color: '#8b949e', fontSize: '11px', marginBottom: '4px' }}>Population</div>
                  <div style={{ fontSize: '20px', fontWeight: 600 }}>{population.length}</div>
                </div>
              </div>

              {/* Control Panel - Compact */}
              <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px', marginBottom: '16px' }}>
                <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
                  {!isRunning ? (
                    <button onClick={startEvolution} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', background: '#238636', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}>
                      <Play size={14} />Start
                    </button>
                  ) : (
                    <>
                      <button onClick={isPaused ? resumeEvolution : pauseEvolution} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', background: '#1f6feb', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}>
                        {isPaused ? <Play size={14} /> : <Pause size={14} />}{isPaused ? 'Resume' : 'Pause'}
                      </button>
                      <button onClick={stopEvolution} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', background: '#da3633', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}>
                        <Square size={14} />Stop
                      </button>
                    </>
                  )}
                  <button onClick={resetEvolution} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', background: '#21262d', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}>
                    <RotateCcw size={14} />Reset
                  </button>
                </div>
                
                {/* Progress Bar - Compact */}
                {isRunning && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '11px' }}>
                      <span>Progress</span>
                      <span>{formatNumber(progress, 0)}%</span>
                    </div>
                    <div style={{ height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #f7931a, #00d4ff)', transition: 'width 0.3s' }}></div>
                    </div>
                  </div>
                )}
              </div>

              {/* Best Strategy - Compact */}
              {bestMutant && (
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <h3 style={{ fontSize: '14px', margin: 0 }}>Best Strategy</h3>
                    <button onClick={() => { setSelectedMutant(bestMutant); setShowExport(true); setActiveTab('export'); }} style={{ display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 10px', background: '#21262d', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}>
                      <Download size={12} />Export
                    </button>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '12px' }}>
                    <div><div style={{ color: '#8b949e', fontSize: '10px' }}>Return</div><div style={{ fontSize: '16px', fontWeight: 600, color: bestMutant.total_return_pct >= 0 ? '#3fb950' : '#f85149' }}>{formatNumber(bestMutant.total_return_pct)}%</div></div>
                    <div><div style={{ color: '#8b949e', fontSize: '10px' }}>Sharpe</div><div style={{ fontSize: '16px', fontWeight: 600 }}>{formatNumber(bestMutant.sharpe_ratio)}</div></div>
                    <div><div style={{ color: '#8b949e', fontSize: '10px' }}>Win%</div><div style={{ fontSize: '16px', fontWeight: 600 }}>{formatNumber(bestMutant.win_rate)}%</div></div>
                    <div><div style={{ color: '#8b949e', fontSize: '10px' }}>Trades</div><div style={{ fontSize: '16px', fontWeight: 600 }}>{bestMutant.total_trades}</div></div>
                    <div><div style={{ color: '#8b949e', fontSize: '10px' }}>Max DD</div><div style={{ fontSize: '16px', fontWeight: 600, color: '#f85149' }}>{formatNumber(bestMutant.max_drawdown)}%</div></div>
                    <div><div style={{ color: '#8b949e', fontSize: '10px' }}>Fitness</div><div style={{ fontSize: '16px', fontWeight: 600, color: '#f7931a' }}>{formatNumber(bestMutant.fitness)}</div></div>
                  </div>
                  <div style={{ marginTop: '10px', padding: '8px', background: '#0d1117', borderRadius: '6px', fontSize: '11px' }}>
                    <span style={{ color: '#8b949e' }}>Ind:</span> <strong>{bestMutant.params.indicator}</strong> | 
                    <span style={{ color: '#8b949e', marginLeft: 8 }}>Dir:</span> <strong>{bestMutant.params.trade_side}</strong> | 
                    <span style={{ color: '#8b949e', marginLeft: 8 }}>Gen:</span> <strong>{bestMutant.generation}</strong>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Population Tab - Compact */}
          {activeTab === 'population' && (
            <div>
              <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Population (Top 20)</h2>
              <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                  <thead style={{ background: '#21262d' }}>
                    <tr>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #30363d' }}>#</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #30363d' }}>Name</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #30363d' }}>Ind</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #30363d' }}>Return</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #30363d' }}>Sharpe</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #30363d' }}>Win%</th>
                      <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #30363d' }}>Fit</th>
                      <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #30363d' }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {population.length === 0 ? (
                      <tr><td colSpan={8} style={{ padding: '24px', textAlign: 'center', color: '#8b949e' }}>Start evolution to see results</td></tr>
                    ) : (
                      population.map((m, idx) => (
                        <tr key={m.id} style={{ borderTop: '1px solid #30363d', background: selectedMutant?.id === m.id ? '#21262d' : 'transparent' }}>
                          <td style={{ padding: '8px 10px' }}>{idx + 1}</td>
                          <td style={{ padding: '8px 10px' }}>{m.name?.slice(0,12)}</td>
                          <td style={{ padding: '8px 10px' }}><span style={{ background: '#21262d', padding: '1px 5px', borderRadius: '3px', fontSize: '10px' }}>{m.params.indicator}</span></td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: m.total_return_pct >= 0 ? '#3fb950' : '#f85149' }}>{formatNumber(m.total_return_pct)}%</td>
                          <td style={{ padding: '8px 10px', textAlign: 'right' }}>{formatNumber(m.sharpe_ratio)}</td>
                          <td style={{ padding: '8px 10px', textAlign: 'right' }}>{formatNumber(m.win_rate)}%</td>
                          <td style={{ padding: '8px 10px', textAlign: 'right', color: '#f7931a' }}>{formatNumber(m.fitness)}</td>
                          <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                            <button onClick={() => setSelectedMutant(m)} style={{ padding: '2px 8px', background: '#21262d', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '3px', cursor: 'pointer', fontSize: '10px' }}>View</button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Mutations Tab - Compact */}
          {activeTab === 'mutations' && (
            <div>
              <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Mutations ({ALL_MUTATIONS.length})</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '8px' }}>
                {ALL_MUTATIONS.map(op => (
                  <div key={op.id} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '10px' }}>
                    <div style={{ fontSize: '16px', marginBottom: '4px' }}>{op.icon}</div>
                    <div style={{ fontWeight: 600, fontSize: '12px', marginBottom: '2px' }}>{op.label}</div>
                    <div style={{ fontSize: '10px', color: '#8b949e', marginBottom: '6px' }}>{op.description?.slice(0,40)}</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '9px', color: '#6b7280', textTransform: 'uppercase', background: '#21262d', padding: '1px 4px', borderRadius: '3px' }}>{op.category}</span>
                      <span style={{ fontSize: '9px', color: '#8b949e' }}>{Math.round(op.probability * 100)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Export Tab - Compact */}
          {activeTab === 'export' && (
            <div>
              <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Export</h2>
              {selectedMutant ? (
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <div>
                      <h3 style={{ fontSize: '14px', margin: '0 0 2px 0' }}>{selectedMutant.name}</h3>
                      <div style={{ fontSize: '11px', color: '#8b949e' }}>Gen {selectedMutant.generation} • Fit: {formatNumber(selectedMutant.fitness)}</div>
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <select value={exportFormat} onChange={e => setExportFormat(e.target.value as any)} style={{ background: '#21262d', border: '1px solid #30363d', borderRadius: '4px', padding: '6px 8px', color: '#e6edf3', fontSize: '11px' }}>
                        <option value="json">JSON</option>
                        <option value="pine">Pine</option>
                        <option value="python">Python</option>
                        <option value="csv">CSV</option>
                      </select>
                      <button onClick={exportCurrent} style={{ display: 'flex', alignItems: 'center', gap: '4px', padding: '6px 12px', background: '#238636', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', fontWeight: 600 }}>
                        <Download size={12} />Export
                      </button>
                    </div>
                  </div>
                  
                  {/* Stats Preview */}
                  <div style={{ background: '#0d1117', borderRadius: '6px', padding: '10px', marginBottom: '10px' }}>
                    <pre style={{ margin: 0, fontSize: '10px', color: '#8b949e', whiteSpace: 'pre-wrap' }}>{generateStatisticsSummary(selectedMutant, selectedMutant.backtest_result)}</pre>
                  </div>
                </div>
              ) : (
                <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '8px', padding: '24px', textAlign: 'center', color: '#8b949e' }}>
                  <Trophy size={32} style={{ marginBottom: 10, opacity: 0.5 }} />
                  <div style={{ fontSize: '13px' }}>Select a strategy to export</div>
                </div>
              )}
            </div>
          )}

          {/* Placeholder tabs - Compact */}
          {['strategies', 'evolution', 'backtest'].includes(activeTab) && (
            <div style={{ textAlign: 'center', padding: '40px 16px', color: '#8b949e' }}>
              <Settings size={32} style={{ marginBottom: 10, opacity: 0.5 }} />
              <div style={{ fontSize: '14px', marginBottom: 4 }}>Coming Soon</div>
              <div style={{ fontSize: '12px' }}>Under development</div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
