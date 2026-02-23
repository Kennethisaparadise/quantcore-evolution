"""
QuantCore - Diversity Dashboard v1.0
Head 29: Visualize population diversity and evolution progress
"""

import random
from typing import Dict, List
from collections import Counter


class DiversityDashboard:
    """Track and visualize population diversity."""
    
    def __init__(self, population_size: int = 100):
        self.population_size = population_size
        self.generation = 0
        self.population = []
        self.diversity_history = []
        self.operator_usage = Counter()
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def generate_population(self):
        """Generate a random population."""
        self.population = []
        strategies = ["RSI", "MACD", "Bollinger", "Supertrend", "Ichimoku", "Fractal"]
        
        for i in range(self.population_size):
            self.population.append({
                "id": i,
                "strategy": random.choice(strategies),
                "fitness": random.uniform(0.1, 1.5),
                "parameters": {
                    "period": random.randint(10, 50),
                    "threshold": random.uniform(0.5, 2.0)
                },
                "mutations": random.randint(0, 10)
            })
        
        # Initial fitness stats
        self.best_fitness_history.append(max(p["fitness"] for p in self.population))
        self.avg_fitness_history.append(sum(p["fitness"] for p in self.population) / len(self.population))
        
        # Calculate initial diversity
        self._calculate_diversity()
    
    def _calculate_diversity(self):
        """Calculate diversity metrics."""
        # Strategy diversity
        strategies = [p["strategy"] for p in self.population]
        strategy_counts = Counter(strategies)
        
        # Shannon entropy
        total = len(self.population)
        shannon = 0
        for count in strategy_counts.values():
            p = count / total
            if p > 0:
                shannon -= p * (p ** 0.5)  # Simplified
        
        # Parameter spread
        all_periods = [p["parameters"]["period"] for p in self.population]
        period_spread = max(all_periods) - min(all_periods)
        
        diversity = {
            "shannon": shannon,
            "unique_strategies": len(strategy_counts),
            "strategy_distribution": dict(strategy_counts),
            "parameter_spread": period_spread,
            "population_size": len(self.population)
        }
        
        self.diversity_history.append(diversity)
        return diversity
    
    def record_operator_usage(self, operator: str):
        """Record operator usage."""
        self.operator_usage[operator] += 1
    
    def evolve(self):
        """Simulate one generation of evolution."""
        self.generation += 1
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        
        # Keep top 20%
        survivors = sorted_pop[:self.population_size // 5]
        
        # Create new offspring
        new_pop = survivors.copy()
        while len(new_pop) < self.population_size:
            parent = random.choice(survivors)
            offspring = {
                "id": len(new_pop),
                "strategy": parent["strategy"],
                "fitness": parent["fitness"] * random.uniform(0.9, 1.1),
                "parameters": parent["parameters"].copy(),
                "mutations": parent["mutations"] + 1
            }
            
            # Slight parameter drift
            if random.random() < 0.3:
                offspring["parameters"]["period"] += random.randint(-5, 5)
                offspring["parameters"]["period"] = max(5, min(100, offspring["parameters"]["period"]))
            
            new_pop.append(offspring)
        
        self.population = new_pop
        
        # Record stats
        self.best_fitness_history.append(max(p["fitness"] for p in self.population))
        self.avg_fitness_history.append(sum(p["fitness"] for p in self.population) / len(self.population))
        
        # Calculate diversity
        self._calculate_diversity()
    
    def get_current_diversity(self) -> Dict:
        """Get current diversity metrics."""
        if not self.diversity_history:
            return {}
        return self.diversity_history[-1]
    
    def get_fitness_history(self) -> Dict:
        """Get fitness history."""
        return {
            "generations": list(range(len(self.best_fitness_history))),
            "best": self.best_fitness_history,
            "avg": self.avg_fitness_history
        }
    
    def get_operator_usage(self) -> Dict:
        """Get operator usage counts."""
        return dict(self.operator_usage)
    
    def generate_html(self) -> str:
        """Generate HTML dashboard."""
        diversity = self.get_current_diversity()
        fitness = self.get_fitness_history()
        operators = self.get_operator_usage()
        
        # Strategy distribution
        strat_dist = diversity.get("strategy_distribution", {})
        strat_bars = ""
        for strat, count in sorted(strat_dist.items(), key=lambda x: -x[1]):
            pct = (count / diversity["population_size"]) * 100
            strat_bars += f"""
            <tr>
                <td>{strat}</td>
                <td>{count}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width:{pct}%"></div>
                    </div>
                </td>
                <td>{pct:.1f}%</td>
            </tr>
            """
        
        # Operator usage
        op_rows = ""
        for op, count in sorted(operators.items(), key=lambda x: -x[1]):
            op_rows += f"""
            <tr>
                <td>{op}</td>
                <td>{count}</td>
            </tr>
            """
        
        # Fitness chart (simple text bars)
        best_fitness = fitness["best"]
        gen_count = len(best_fitness)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Diversity Dashboard - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #3fb950;
            margin-bottom: 20px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
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
            color: #3fb950;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .charts {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .chart {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 20px;
        }}
        .chart-title {{
            color: #58a6ff;
            font-size: 14px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #21262d;
            padding: 10px;
            text-align: left;
            color: #8b949e;
            font-size: 10px;
            text-transform: uppercase;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #21262d;
            font-size: 12px;
        }}
        .bar-container {{
            width: 100px;
            height: 10px;
            background: #21262d;
            border-radius: 5px;
            overflow: hidden;
        }}
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, #3fb950, #58a6ff);
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">🧬 DIVERSITY DASHBOARD</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{diversity.get('unique_strategies', 0)}</div>
            <div class="stat-label">Unique Strategies</div>
        </div>
        <div class="stat">
            <div class="stat-value">{diversity.get('population_size', 0)}</div>
            <div class="stat-label">Population</div>
        </div>
        <div class="stat">
            <div class="stat-value">{diversity.get('shannon', 0):.2f}</div>
            <div class="stat-label">Diversity Index</div>
        </div>
        <div class="stat">
            <div class="stat-value">Gen {self.generation}</div>
            <div class="stat-label">Generation</div>
        </div>
    </div>
    
    <div class="charts">
        <div class="chart">
            <div class="chart-title">Strategy Distribution</div>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Count</th>
                        <th>Visual</th>
                        <th>%</th>
                    </tr>
                </thead>
                <tbody>
                    {strat_bars}
                </tbody>
            </table>
        </div>
        
        <div class="chart">
            <div class="chart-title">Operator Usage</div>
            <table>
                <thead>
                    <tr>
                        <th>Operator</th>
                        <th>Uses</th>
                    </tr>
                </thead>
                <tbody>
                    {op_rows or '<tr><td colspan="2">No data yet</td></tr>'}
                </tbody>
            </table>
        </div>
        
        <div class="chart">
            <div class="chart-title">Fitness History</div>
            <div style="font-size:11px;color:#8b949e;">
                Generation: {gen_count}<br>
                Best Fitness: {best_fitness[-1] if best_fitness else 0:.3f}<br>
                Avg Fitness: {fitness["avg"][-1] if fitness["avg"] else 0:.3f}
            </div>
        </div>
        
        <div class="chart">
            <div class="chart-title">Quick Stats</div>
            <div style="font-size:11px;color:#8b949e;line-height:1.8;">
                Parameter Spread: {diversity.get('parameter_spread', 0)}<br>
                Best Ever: {max(best_fitness) if best_fitness else 0:.3f}<br>
                Worst Ever: {min(best_fitness) if best_fitness else 0:.3f}<br>
                Trend: {'📈 Improving' if len(best_fitness) > 1 and best_fitness[-1] > best_fitness[0] else '📉 Declining'}
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("DIVERSITY DASHBOARD TEST")
    print("=" * 50)
    
    dashboard = DiversityDashboard(50)
    dashboard.generate_population()
    
    # Simulate evolution
    for i in range(10):
        # Random operator usage
        ops = ["semantic_alignment", "adaptive_rate", "differential_evolution", "decay_schedule"]
        dashboard.record_operator_usage(random.choice(ops))
        dashboard.evolve()
    
    diversity = dashboard.get_current_diversity()
    print(f"\nGeneration: {dashboard.generation}")
    print(f"Unique strategies: {diversity['unique_strategies']}")
    print(f"Diversity index: {diversity['shannon']:.3f}")
    print(f"Population: {diversity['population_size']}")
    
    # Save HTML
    html = dashboard.generate_html()
    with open("diversity_dashboard.html", "w") as f:
        f.write(html)
    
    print("\nSaved to diversity_dashboard.html")
    print("OK!")
