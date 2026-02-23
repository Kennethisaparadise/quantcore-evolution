"""
QuantCore - Mutation Harvester v1.0
Head 27: Harvest and implement novel mutation operators from research
"""

import random
import json
from typing import Dict, List, Callable
from dataclasses import dataclass


@dataclass
class MutationOperator:
    """A mutation operator for the genetic algorithm."""
    name: str
    description: str
    category: str  # core, adaptive, semantic, tree, differential
    function: Callable
    parameters: Dict


class MutationHarvester:
    """Harvest and manage mutation operators."""
    
    # Core mutations (existing)
    CORE_MUTATIONS = [
        "parameter_tweak",
        "indicator_swap",
        "threshold_shift",
        "timeframe_change",
        "logic_inversion",
    ]
    
    # New operators from research
    NEW_OPERATORS = []
    
    def __init__(self):
        self.operators: Dict[str, MutationOperator] = {}
        self.operator_history: List[Dict] = []
        self._register_operators()
    
    def _register_operators(self):
        """Register all mutation operators."""
        
        # 1. Semantic Alignment Mutation
        self.operators["semantic_alignment"] = MutationOperator(
            name="semantic_alignment",
            description="Mutate paired parameters together (entry/exit)",
            category="semantic",
            function=self._semantic_mutate,
            parameters={
                "semantic_weight": {"min": 0, "max": 1, "default": 0.5}
            }
        )
        
        # 2. Adaptive Mutation Rate
        self.operators["adaptive_rate"] = MutationOperator(
            name="adaptive_rate",
            description="Mutation rate adjusts based on population diversity",
            category="adaptive",
            function=self._adaptive_mutate,
            parameters={
                "base_rate": {"min": 0.01, "max": 0.3, "default": 0.1},
                "diversity_threshold": {"min": 0.1, "max": 0.5, "default": 0.3}
            }
        )
        
        # 3. Split Population Mutation
        self.operators["split_population"] = MutationOperator(
            name="split_population",
            description="Two sub-populations with different mutation rates",
            category="adaptive",
            function=self._split_population_mutate,
            parameters={
                "conservative_rate": {"min": 0.01, "max": 0.1, "default": 0.05},
                "exploratory_rate": {"min": 0.15, "max": 0.4, "default": 0.25},
                "migration_interval": {"min": 5, "max": 20, "default": 10}
            }
        )
        
        # 4. Differential Evolution Mutation
        self.operators["differential_evolution"] = MutationOperator(
            name="differential_evolution",
            description="Use vector differences: mutant = x1 + F*(x2 - x3)",
            category="differential",
            function=self._differential_mutate,
            parameters={
                "F_factor": {"min": 0.4, "max": 1.0, "default": 0.6}
            }
        )
        
        # 5. Tree Depth-Aware Mutation
        self.operators["depth_aware"] = MutationOperator(
            name="depth_aware",
            description="Respect depth limits when mutating expression trees",
            category="tree",
            function=self._depth_aware_mutate,
            parameters={
                "max_depth": {"min": 3, "max": 8, "default": 5},
                "prune_probability": {"min": 0, "max": 1, "default": 0.3}
            }
        )
        
        # 6. Mutation Decay Schedule
        self.operators["decay_schedule"] = MutationOperator(
            name="decay_schedule",
            description="Mutation scale decreases over generations",
            category="adaptive",
            function=self._decay_mutate,
            parameters={
                "initial_scale": {"min": 0.5, "max": 1.5, "default": 1.0},
                "decay_rate": {"min": 0.9, "max": 1.0, "default": 0.98}
            }
        )
        
        # 7. Attempt-Based Mutation
        self.operators["attempt_based"] = MutationOperator(
            name="attempt_based",
            description="Try multiple times before giving up on valid mutation",
            category="core",
            function=self._attempt_mutate,
            parameters={
                "max_attempts": {"min": 1, "max": 10, "default": 3}
            }
        )
        
        # 8. Cooldown Mutation
        self.operators["cooldown"] = MutationOperator(
            name="cooldown",
            description="Skip mutation if parent was recently mutated",
            category="adaptive",
            function=self._cooldown_mutate,
            parameters={
                "cooldown_generations": {"min": 1, "max": 5, "default": 2}
            }
        )
    
    # Mutation implementations
    def _semantic_mutate(self, strategy: Dict, params: Dict, **kwargs) -> Dict:
        """Mutate paired parameters together."""
        semantic_weight = params.get("semantic_weight", 0.5)
        
        # Example: entry/exit threshold pairs
        if "entry_threshold" in strategy and "exit_threshold" in strategy:
            if random.random() < semantic_weight:
                factor = random.uniform(0.8, 1.2)
                strategy["entry_threshold"] *= factor
                strategy["exit_threshold"] *= factor
            else:
                strategy["entry_threshold"] *= random.uniform(0.8, 1.2)
        
        return strategy
    
    def _adaptive_mutate(self, strategy: Dict, params: Dict, diversity: float = 0.5) -> Dict:
        """Adjust mutation rate based on diversity."""
        base_rate = params.get("base_rate", 0.1)
        threshold = params.get("diversity_threshold", 0.3)
        
        # When diversity drops, increase mutation
        if diversity < threshold:
            mutation_rate = base_rate * 2
        else:
            mutation_rate = base_rate
        
        # Apply mutation with adjusted rate
        if random.random() < mutation_rate:
            strategy["parameters"] = strategy.get("parameters", {})
            for key in strategy["parameters"]:
                if random.random() < 0.3:  # 30% of parameters
                    strategy["parameters"][key] *= random.uniform(0.9, 1.1)
        
        return strategy
    
    def _split_population_mutate(self, strategy: Dict, params: Dict, 
                                  in_exploratory: bool = False) -> Dict:
        """Use different mutation rates for different sub-populations."""
        if in_exploratory:
            rate = params.get("exploratory_rate", 0.25)
            scale = 0.3  # Larger changes
        else:
            rate = params.get("conservative_rate", 0.05)
            scale = 0.1  # Small tweaks
        
        if random.random() < rate:
            strategy["parameters"] = strategy.get("parameters", {})
            for key in strategy["parameters"]:
                if random.random() < 0.5:
                    strategy["parameters"][key] *= random.uniform(1 - scale, 1 + scale)
        
        return strategy
    
    def _differential_mutate(self, strategy: Dict, params: Dict,
                             others: List[Dict] = None) -> Dict:
        """Differential evolution: mutant = x1 + F*(x2 - x3)"""
        F = params.get("F_factor", 0.6)
        
        if others and len(others) >= 2:
            x1, x2, x3 = random.sample(others, 3)
            
            # Create mutant for numeric parameters
            for key in strategy.get("parameters", {}):
                if key in x1.get("parameters", {}) and key in x2.get("parameters", {}):
                    val1 = x1["parameters"].get(key, 0)
                    val2 = x2["parameters"].get(key, 0)
                    val3 = x3["parameters"].get(key, 0)
                    
                    mutant_val = val1 + F * (val2 - val3)
                    strategy["parameters"][key] = mutant_val
        
        return strategy
    
    def _depth_aware_mutate(self, strategy: Dict, params: Dict) -> Dict:
        """Respect depth limits when mutating trees."""
        max_depth = params.get("max_depth", 5)
        prune_prob = params.get("prune_probability", 0.3)
        
        # Simulate tree depth
        current_depth = strategy.get("tree_depth", 3)
        
        if current_depth >= max_depth:
            # Prune instead of adding
            if random.random() < prune_prob:
                strategy["pruned"] = True
        else:
            # Normal mutation
            strategy["tree_depth"] = current_depth + 1
        
        return strategy
    
    def _decay_mutate(self, strategy: Dict, params: Dict, generation: int = 0) -> Dict:
        """Mutation scale decreases over generations."""
        initial = params.get("initial_scale", 1.0)
        decay = params.get("decay_rate", 0.98)
        
        scale = initial * (decay ** generation)
        
        strategy["parameters"] = strategy.get("parameters", {})
        for key in strategy["parameters"]:
            if random.random() < 0.3:
                strategy["parameters"][key] *= random.uniform(1 - scale, 1 + scale)
        
        return strategy
    
    def _attempt_mutate(self, strategy: Dict, params: Dict) -> Dict:
        """Try multiple times to find valid mutation."""
        max_attempts = params.get("max_attempts", 3)
        
        for attempt in range(max_attempts):
            mutated = strategy.copy()
            mutated["parameters"] = mutated.get("parameters", {}).copy()
            
            for key in mutated["parameters"]:
                mutated["parameters"][key] *= random.uniform(0.8, 1.2)
            
            # Validate (placeholder - would check strategy validity)
            if self._is_valid(mutated):
                return mutated
        
        return strategy  # Return original if all attempts fail
    
    def _cooldown_mutate(self, strategy: Dict, params: Dict) -> Dict:
        """Skip mutation if parent was recently mutated."""
        cooldown = params.get("cooldown_generations", 2)
        
        generations_since_mutation = strategy.get("generations_since_mutation", 999)
        
        if generations_since_mutation < cooldown:
            # Still in cooldown
            strategy["mutated"] = False
        else:
            # Normal mutation
            strategy["mutated"] = True
            strategy["generations_since_mutation"] = 0
        
        strategy["generations_since_mutation"] = generations_since_mutation + 1
        return strategy
    
    def _is_valid(self, strategy: Dict) -> bool:
        """Validate mutated strategy."""
        # Basic validation
        params = strategy.get("parameters", {})
        for key, val in params.items():
            if not isinstance(val, (int, float)):
                return False
            if val < 0 or val > 10000:  # Reasonable bounds
                return False
        return True
    
    def mutate(self, strategy: Dict, operator: str, params: Dict = None,
               **context) -> Dict:
        """Apply a mutation operator to a strategy."""
        if operator not in self.operators:
            raise ValueError(f"Unknown operator: {operator}")
        
        op = self.operators[operator]
        p = params or {}
        
        # Get context parameters
        context_params = {
            "diversity": context.get("diversity", 0.5),
            "generation": context.get("generation", 0),
            "in_exploratory": context.get("in_exploratory", False),
            "others": context.get("others", [])
        }
        
        # Apply mutation
        mutated = op.function(strategy.copy(), p, **context_params)
        
        # Record history
        self.operator_history.append({
            "operator": operator,
            "timestamp": "now",
            "success": self._is_valid(mutated)
        })
        
        return mutated
    
    def get_operators_by_category(self, category: str) -> List[str]:
        """Get all operators in a category."""
        return [name for name, op in self.operators.items() 
                if op.category == category]
    
    def get_operator_info(self) -> List[Dict]:
        """Get info about all operators."""
        return [
            {
                "name": op.name,
                "description": op.description,
                "category": op.category,
                "parameters": op.parameters
            }
            for op in self.operators.values()
        ]
    
    def generate_html(self) -> str:
        """Generate HTML report."""
        operators = self.get_operator_info()
        
        # Group by category
        categories = {}
        for op in operators:
            cat = op["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(op)
        
        category_html = ""
        for cat, ops in categories.items():
            ops_html = ""
            for op in ops:
                params = ", ".join(f"{k}: {v['default']}" 
                                 for k, v in op["parameters"].items())
                ops_html += f"""
                <tr>
                    <td>{op['name']}</td>
                    <td>{op['description']}</td>
                    <td>{params}</td>
                </tr>
                """
            
            category_html += f"""
            <div class="category">
                <h3 style="color:#{self._cat_color(cat)}">{cat.upper()}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Operator</th>
                            <th>Description</th>
                            <th>Parameters</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ops_html}
                    </tbody>
                </table>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mutation Harvester - QuantCore</title>
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
            font-size: 32px;
            color: #a371f7;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
        }}
        .category {{
            margin-bottom: 30px;
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
            font-size: 11px;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #21262d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">🧬 MUTATION HARVESTER</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(self.operators)}</div>
            <div class="stat-label">Operators</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(categories)}</div>
            <div class="stat-label">Categories</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(self.operator_history)}</div>
            <div class="stat-label">Mutations Applied</div>
        </div>
    </div>
    
    {category_html}
</body>
</html>
"""
        return html
    
    def _cat_color(self, cat: str) -> str:
        colors = {
            "core": "58a6ff",
            "adaptive": "3fb950",
            "semantic": "f7931a",
            "tree": "a371f7",
            "differential": "f85149"
        }
        return colors.get(cat, "8b949e")


if __name__ == "__main__":
    print("=" * 50)
    print("MUTATION HARVESTER TEST")
    print("=" * 50)
    
    harvester = MutationHarvester()
    
    print(f"\nOperators registered: {len(harvester.operators)}")
    
    # Show by category
    for cat in ["core", "adaptive", "semantic", "tree", "differential"]:
        ops = harvester.get_operators_by_category(cat)
        if ops:
            print(f"  {cat}: {', '.join(ops)}")
    
    # Test mutation
    test_strategy = {
        "name": "test",
        "parameters": {"rsi_length": 14, "rsi_oversold": 30},
        "tree_depth": 3,
        "generations_since_mutation": 10
    }
    
    mutated = harvester.mutate(test_strategy, "semantic_alignment", 
                               {"semantic_weight": 0.5})
    print(f"\nOriginal: {test_strategy['parameters']}")
    print(f"Mutated:  {mutated['parameters']}")
    
    # Generate HTML
    html = harvester.generate_html()
    with open("mutation_harvester.html", "w") as f:
        f.write(html)
    
    print("\nSaved to mutation_harvester.html")
    print("OK!")
