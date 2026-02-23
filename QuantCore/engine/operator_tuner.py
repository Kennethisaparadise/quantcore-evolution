"""
QuantCore - Operator Tuner v1.0
Head 28: Automatically tune mutation operator weights based on performance
"""

import random
from typing import Dict, List
from collections import defaultdict


class OperatorTuner:
    """Auto-tune mutation operator weights based on performance."""
    
    def __init__(self):
        # Default weights for each operator
        self.weights = {
            "semantic_alignment": 0.10,
            "adaptive_rate": 0.15,
            "split_population": 0.12,
            "differential_evolution": 0.15,
            "depth_aware": 0.08,
            "decay_schedule": 0.10,
            "attempt_based": 0.15,
            "cooldown": 0.15,
        }
        
        # Performance tracking
        self.operator_stats = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "avg_fitness_gain": 0,
            "total_gain": 0
        })
        
        # Tuning parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.min_weight = 0.02
        self.max_weight = 0.40
    
    def select_operator(self) -> str:
        """Select operator based on weights (roulette wheel)."""
        total = sum(self.weights.values())
        r = random.random() * total
        
        cumulative = 0
        for op, weight in self.weights.items():
            cumulative += weight
            if r <= cumulative:
                return op
        
        return list(self.weights.keys())[0]
    
    def record_result(self, operator: str, fitness_before: float, fitness_after: float):
        """Record the result of using an operator."""
        stats = self.operator_stats[operator]
        gain = fitness_after - fitness_before
        
        stats["attempts"] += 1
        if gain > 0:
            stats["successes"] += 1
        
        # Update running average
        n = stats["attempts"]
        stats["avg_fitness_gain"] = (
            (stats["avg_fitness_gain"] * (n - 1) + gain) / n
        )
        stats["total_gain"] += gain
    
    def tune_weights(self):
        """Adjust operator weights based on performance."""
        # Get average performance across all operators
        all_gains = [s["avg_fitness_gain"] for s in self.operator_stats.values()]
        
        if not all_gains or max(all_gains) == min(all_gains) == 0:
            return  # No data yet
        
        # Adjust weights based on relative performance
        for op, stats in self.operator_stats.items():
            if stats["attempts"] == 0:
                continue
            
            # Calculate performance score
            perf = stats["avg_fitness_gain"]
            win_rate = stats["successes"] / max(1, stats["attempts"])
            
            # Combined score
            score = (perf * 0.6) + (win_rate * 0.4)
            
            # Adjust weight
            if score > 0:
                self.weights[op] = min(
                    self.max_weight,
                    self.weights[op] * (1 + self.learning_rate)
                )
            else:
                self.weights[op] = max(
                    self.min_weight,
                    self.weights[op] * (1 - self.learning_rate)
                )
        
        # Normalize weights
        self._normalize_weights()
        
        # Decay learning rate
        self.learning_rate *= self.decay_factor
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        for op in self.weights:
            self.weights[op] /= total
    
    def get_best_operator(self) -> str:
        """Get the currently best-performing operator."""
        best = None
        best_score = float('-inf')
        
        for op, stats in self.operator_stats.items():
            if stats["attempts"] == 0:
                continue
            
            score = stats["avg_fitness_gain"] * stats["successes"] / stats["attempts"]
            if score > best_score:
                best_score = score
                best = op
        
        return best or "adaptive_rate"
    
    def get_weights(self) -> Dict[str, float]:
        """Get current operator weights."""
        return self.weights.copy()
    
    def get_stats(self) -> Dict:
        """Get detailed operator statistics."""
        return dict(self.operator_stats)
    
    def generate_html(self) -> str:
        """Generate HTML dashboard."""
        weights = self.weights
        stats = self.get_stats()
        best = self.get_best_operator()
        
        # Weight bars
        weight_rows = ""
        for op, weight in sorted(weights.items(), key=lambda x: -x[1]):
            stat = stats.get(op, {"attempts": 0, "successes": 0, "avg_fitness_gain": 0})
            is_best = "🏆" if op == best else ""
            
            win_rate = (stat["successes"] / max(1, stat["attempts"])) * 100 if stat["attempts"] > 0 else 0
            
            weight_rows += f"""
            <tr>
                <td>{op} {is_best}</td>
                <td>{weight*100:.1f}%</td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width:{weight*100}%"></div>
                    </div>
                </td>
                <td>{stat['attempts']}</td>
                <td>{win_rate:.0f}%</td>
                <td>{stat['avg_fitness_gain']:+.2f}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Operator Tuner - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #f7931a;
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
            font-size: 32px;
            color: #f7931a;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
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
            width: 100px;
            height: 12px;
            background: #21262d;
            border-radius: 6px;
            overflow: hidden;
        }}
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, #f7931a, #e3b341);
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
    <div class="header">🎛️ OPERATOR TUNER</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(self.weights)}</div>
            <div class="stat-label">Operators</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(s['attempts'] for s in stats.values())}</div>
            <div class="stat-label">Total Attempts</div>
        </div>
        <div class="stat">
            <div class="stat-value">{best}</div>
            <div class="stat-label">Best Operator</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self.learning_rate:.3f}</div>
            <div class="stat-label">Learning Rate</div>
        </div>
    </div>
    
    <h3 style="color:#58a6ff">Operator Weights</h3>
    <table>
        <thead>
            <tr>
                <th>Operator</th>
                <th>Weight</th>
                <th>Visual</th>
                <th>Attempts</th>
                <th>Win Rate</th>
                <th>Avg Gain</th>
            </tr>
        </thead>
        <tbody>
            {weight_rows}
        </tbody>
    </table>
    
    <div class="controls">
        <button onclick="tune()">Tune Weights</button>
        <button onclick="reset()">Reset</button>
    </div>
    
    <script>
        function tune() {{
            fetch('/tune', {{method: 'POST'}}).then(r => location.reload());
        }}
        function reset() {{
            fetch('/reset', {{method: 'POST'}}).then(r => location.reload());
        }}
    </script>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    print("=" * 50)
    print("OPERATOR TUNER TEST")
    print("=" * 50)
    
    tuner = OperatorTuner()
    
    # Simulate usage
    operators = list(tuner.weights.keys())
    for i in range(100):
        op = tuner.select_operator()
        
        # Simulate fitness change
        before = random.uniform(0.5, 1.5)
        after = before + random.uniform(-0.1, 0.2)
        
        tuner.record_result(op, before, after)
    
    # Tune weights
    tuner.tune_weights()
    
    print("\nOperator Weights:")
    for op, w in sorted(tuner.get_weights().items(), key=lambda x: -x[1]):
        print(f"  {op}: {w*100:.1f}%")
    
    print(f"\nBest: {tuner.get_best_operator()}")
    
    # Save HTML
    html = tuner.generate_html()
    with open("operator_tuner.html", "w") as f:
        f.write(html)
    
    print("\nSaved to operator_tuner.html")
    print("OK!")
