"""
QuantCore - Meta-Fitness Engine v1.0
Head 30: Evolve the fitness function itself
"""

import random
from typing import Dict, List, Callable
from dataclasses import dataclass


@dataclass
class FitnessComponent:
    """A component of the fitness function."""
    name: str
    weight: float
    function: Callable
    description: str


class MetaFitnessEngine:
    """
    Meta-evolutionary fitness function.
    Evolves what "good" means for strategy selection.
    """
    
    # Default fitness components
    DEFAULT_COMPONENTS = [
        "sharpe_ratio",
        "total_return",
        "win_rate",
        "max_drawdown",
        "profit_factor",
        "calmar_ratio",
    ]
    
    def __init__(self):
        # Fitness components with weights
        self.components: Dict[str, FitnessComponent] = {}
        self.component_weights: Dict[str, float] = {}
        
        # Evolution history
        self.weight_history: List[Dict] = []
        self.best_fitness_history: List[float] = []
        
        # Register default components
        self._register_default_components()
        
        # Current fitness function
        self.fitness_function = self._build_fitness_function()
    
    def _register_default_components(self):
        """Register default fitness components."""
        
        def sharpe_ratio(returns: List[float]) -> float:
            if not returns or len(returns) < 2:
                return 0
            mean = sum(returns) / len(returns)
            std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5
            if std == 0:
                return 0
            return mean / std * (252 ** 0.5)  # Annualized
        
        def total_return(trades: List[Dict]) -> float:
            if not trades:
                return 0
            return sum(t.get("pnl", 0) for t in trades) / 10000
        
        def win_rate(trades: List[Dict]) -> float:
            if not trades:
                return 0
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            return wins / len(trades)
        
        def max_drawdown(trades, equity_curve: List[float]) -> float:
            if not equity_curve:
                return 0
            peak = equity_curve[0]
            max_dd = 0
            for e in equity_curve:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            return -max_dd  # Negative is bad
        
        def profit_factor(trades: List[Dict]) -> float:
            if not trades:
                return 0
            gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
            gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
            if gross_loss == 0:
                return 0
            return gross_profit / gross_loss
        
        def calmar_ratio(trades: List[Dict], equity_curve: List[float]) -> float:
            if not trades or not equity_curve:
                return 0
            total_ret = sum(t.get("pnl", 0) for t in trades)
            max_dd = max_drawdown(trades, equity_curve)
            if max_dd == 0:
                return 0
            return total_ret / abs(max_dd)
        
        # Register components
        self.components["sharpe_ratio"] = FitnessComponent(
            name="sharpe_ratio",
            weight=0.25,
            function=sharpe_ratio,
            description="Risk-adjusted return"
        )
        
        self.components["total_return"] = FitnessComponent(
            name="total_return",
            weight=0.25,
            function=total_return,
            description="Absolute return"
        )
        
        self.components["win_rate"] = FitnessComponent(
            name="win_rate",
            weight=0.15,
            function=win_rate,
            description="Percentage of winning trades"
        )
        
        self.components["max_drawdown"] = FitnessComponent(
            name="max_drawdown",
            weight=0.20,
            function=max_drawdown,
            description="Largest peak-to-trough"
        )
        
        self.components["profit_factor"] = FitnessComponent(
            name="profit_factor",
            weight=0.10,
            function=profit_factor,
            description="Gross profit / gross loss"
        )
        
        self.components["calmar_ratio"] = FitnessComponent(
            name="calmar_ratio",
            weight=0.05,
            function=calmar_ratio,
            description="Return / max drawdown"
        )
        
        # Set initial weights
        for name, comp in self.components.items():
            self.component_weights[name] = comp.weight
    
    def _build_fitness_function(self) -> Callable:
        """Build the composite fitness function."""
        def fitness(strategy_data: Dict) -> float:
            trades = strategy_data.get("trades", [])
            equity_curve = strategy_data.get("equity_curve", [])
            returns = [t.get("pnl", 0) / 10000 for t in trades]
            
            total = 0
            for name, weight in self.component_weights.items():
                if name in self.components:
                    comp = self.components[name]
                    
                    # Call with appropriate arguments
                    if name == "sharpe_ratio":
                        score = comp.function(returns)
                    elif name in ["max_drawdown", "calmar_ratio"]:
                        score = comp.function(trades, equity_curve)
                    else:
                        score = comp.function(trades)
                    
                    total += weight * score
            
            return total
        
        return fitness
    
    def calculate_fitness(self, strategy_data: Dict) -> float:
        """Calculate fitness for a strategy."""
        return self.fitness_function(strategy_data)
    
    def mutate_weights(self, mutation_rate: float = 0.1):
        """Mutate the fitness component weights."""
        for name in self.component_weights:
            if random.random() < mutation_rate:
                # Adjust weight
                change = random.uniform(-0.05, 0.05)
                self.component_weights[name] = max(0.01, min(0.5, 
                    self.component_weights[name] + change))
        
        # Normalize weights
        total = sum(self.component_weights.values())
        for name in self.component_weights:
            self.component_weights[name] /= total
        
        # Rebuild function
        self.fitness_function = self._build_fitness_function()
        
        # Record history
        self.weight_history.append(self.component_weights.copy())
    
    def optimize_weights(self, validation_results: List[Dict]):
        """
        Optimize weights based on validation results.
        validation_results: [{"weights": {...}, "fitness": float}]
        """
        if not validation_results:
            return
        
        # Find best performing weights
        best = max(validation_results, key=lambda x: x["fitness"])
        
        # Blend towards best
        blend_factor = 0.3
        for name in self.component_weights:
            if name in best["weights"]:
                self.component_weights[name] = (
                    (1 - blend_factor) * self.component_weights[name] +
                    blend_factor * best["weights"][name]
                )
        
        # Normalize
        total = sum(self.component_weights.values())
        for name in self.component_weights:
            self.component_weights[name] /= total
        
        # Rebuild function
        self.fitness_function = self._build_fitness_function()
        
        # Record history
        self.weight_history.append(self.component_weights.copy())
    
    def get_weights(self) -> Dict[str, float]:
        """Get current component weights."""
        return self.component_weights.copy()
    
    def get_components(self) -> List[Dict]:
        """Get component info."""
        return [
            {
                "name": name,
                "weight": self.component_weights.get(name, 0),
                "description": comp.description
            }
            for name, comp in self.components.items()
        ]
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        components = self.get_components()
        
        # Sort by weight
        sorted_comps = sorted(components, key=lambda x: -x["weight"])
        
        # Weight bars
        weight_rows = ""
        for c in sorted_comps:
            pct = c["weight"] * 100
            weight_rows += f"""
            <tr>
                <td>{c['name']}</td>
                <td>{c['description']}</td>
                <td>{pct:.1f}%</td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width:{pct}%"></div>
                    </div>
                </td>
            </tr>
            """
        
        # Evolution history chart (simple)
        history_len = len(self.weight_history)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Meta-Fitness Engine - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #a371f7;
            margin-bottom: 20px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat {{
            background: #161b22;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }}
        .stat-value {{
            font-size: 28px;
            color: #a371f7;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #21262d;
            padding: 12px;
            text-align: left;
            color: #8b949e;
            font-size: 11px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #21262d;
        }}
        .bar-container {{
            width: 150px;
            height: 12px;
            background: #21262d;
            border-radius: 6px;
            overflow: hidden;
        }}
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, #a371f7, #58a6ff);
            border-radius: 6px;
        }}
        .controls {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }}
        button {{
            background: #238636;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-family: monospace;
        }}
        button:hover {{
            background: #2ea043;
        }}
    </style>
</head>
<body>
    <div class="header">🧬 META-FITNESS ENGINE</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(self.components)}</div>
            <div class="stat-label">Components</div>
        </div>
        <div class="stat">
            <div class="stat-value">{history_len}</div>
            <div class="stat-label">Evolutions</div>
        </div>
        <div class="stat">
            <div class="stat-value">Active</div>
            <div class="stat-label">Status</div>
        </div>
    </div>
    
    <h3 style="color:#58a6ff">Fitness Components</h3>
    <table>
        <thead>
            <tr>
                <th>Component</th>
                <th>Description</th>
                <th>Weight</th>
                <th>Visual</th>
            </tr>
        </thead>
        <tbody>
            {weight_rows}
        </tbody>
    </table>
    
    <div class="controls">
        <button onclick="mutate()">Mutate Weights</button>
    </div>
    
    <script>
        function mutate() {{
            fetch('/mutate', {{method: 'POST'}}).then(r => location.reload());
        }}
    </script>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("META-FITNESS ENGINE TEST")
    print("=" * 50)
    
    engine = MetaFitnessEngine()
    
    print(f"\nComponents: {len(engine.components)}")
    print("\nInitial Weights:")
    for name, w in engine.get_weights().items():
        print(f"  {name}: {w*100:.1f}%")
    
    # Test fitness calculation
    test_data = {
        "trades": [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 200},
            {"pnl": 150}, {"pnl": -30}, {"pnl": 80}
        ],
        "equity_curve": [10000, 10100, 10050, 10250, 10400, 10370, 10450]
    }
    
    fitness = engine.calculate_fitness(test_data)
    print(f"\nTest Fitness: {fitness:.3f}")
    
    # Mutate weights
    engine.mutate_weights(0.5)
    print("\nAfter Mutation:")
    for name, w in engine.get_weights().items():
        print(f"  {name}: {w*100:.1f}%")
    
    # Save HTML
    html = engine.generate_html()
    with open("meta_fitness.html", "w") as f:
        f.write(html)
    
    print("\nSaved to meta_fitness.html")
    print("OK!")
