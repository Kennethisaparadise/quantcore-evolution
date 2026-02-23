"""
QuantCore - Price Prediction Harvester v1.0
Head 35: Harvest and implement price prediction models
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PredictionModel:
    """A price prediction model."""
    name: str
    source: str
    architecture: str  # LSTM, GRU, CNN, SVM, etc.
    horizon: str  # short, medium, long
    approach: str  # statistical, ml, hybrid
    accuracy: float  # Reported accuracy
    data_requirements: List[str]
    features: List[str]


class PricePredictionHarvester:
    """Harvest and manage price prediction models."""
    
    # Known models from research
    KNOWN_MODELS = [
        {
            "name": "LSTM with Attention",
            "source": "arXiv",
            "architecture": "LSTM",
            "horizon": "short",
            "approach": "ml",
            "accuracy": 0.85,
            "data_requirements": ["OHLCV", "volume"],
            "features": ["sequence", "attention", "bidirectional"]
        },
        {
            "name": "GRU with Transfer Learning",
            "source": "arXiv",
            "architecture": "GRU",
            "horizon": "medium",
            "approach": "ml",
            "accuracy": 0.82,
            "data_requirements": ["OHLCV"],
            "features": ["transfer", "fine_tuning"]
        },
        {
            "name": "CNN-LSTM Hybrid",
            "source": "IEEE",
            "architecture": "CNN-LSTM",
            "horizon": "short",
            "approach": "hybrid",
            "accuracy": 0.88,
            "data_requirements": ["OHLCV", "volume", "order_book"],
            "features": ["convolution", "feature_extraction"]
        },
        {
            "name": "Wavelet-ELM",
            "source": "ScienceDirect",
            "architecture": "ELM",
            "horizon": "medium",
            "approach": "hybrid",
            "accuracy": 0.79,
            "data_requirements": ["OHLCV", "time"],
            "features": ["wavelet", "decomposition", "extreme_learning"]
        },
        {
            "name": "BiLSTM with MRMR",
            "source": "arXiv",
            "architecture": "BiLSTM",
            "horizon": "short",
            "approach": "ml",
            "accuracy": 0.86,
            "data_requirements": ["OHLCV", "sentiment"],
            "features": ["bidirectional", "feature_selection"]
        },
        {
            "name": "SVM with Optimization",
            "source": "IEEEXplore",
            "architecture": "SVM",
            "horizon": "short",
            "approach": "ml",
            "accuracy": 0.75,
            "data_requirements": ["OHLCV", "indicators"],
            "features": ["kernel", "hyperparameter_tuning"]
        },
        {
            "name": "ARIMA-GARCH",
            "source": "Academic",
            "architecture": "ARIMA-GARCH",
            "horizon": "medium",
            "approach": "statistical",
            "accuracy": 0.68,
            "data_requirements": ["OHLCV"],
            "features": ["volatility", "time_series"]
        },
        {
            "name": "Prophet",
            "source": "Facebook",
            "architecture": "Prophet",
            "horizon": "long",
            "approach": "statistical",
            "accuracy": 0.72,
            "data_requirements": ["OHLCV", "dates"],
            "features": ["seasonality", "trend", "holidays"]
        },
        {
            "name": "Transformer Encoder",
            "source": "arXiv",
            "architecture": "Transformer",
            "horizon": "short",
            "approach": "ml",
            "accuracy": 0.89,
            "data_requirements": ["OHLCV", "volume"],
            "features": ["attention", "positional_encoding", "self_attention"]
        },
        {
            "name": "XGBoost Ensemble",
            "source": "Kaggle",
            "architecture": "XGBoost",
            "horizon": "short",
            "approach": "ml",
            "accuracy": 0.81,
            "data_requirements": ["OHLCV", "indicators"],
            "features": ["ensemble", "gradient_boosting", "feature_importance"]
        },
    ]
    
    def __init__(self):
        self.models: List[PredictionModel] = []
        self._load_models()
        self.active_model = None
    
    def _load_models(self):
        """Load known prediction models."""
        for m in self.KNOWN_MODELS:
            model = PredictionModel(
                name=m["name"],
                source=m["source"],
                architecture=m["architecture"],
                horizon=m["horizon"],
                approach=m["approach"],
                accuracy=m["accuracy"],
                data_requirements=m["data_requirements"],
                features=m["features"]
            )
            self.models.append(model)
    
    def get_model(self, index: int) -> PredictionModel:
        """Get model by index."""
        return self.models[index % len(self.models)]
    
    def get_models_by_approach(self, approach: str) -> List[PredictionModel]:
        """Get models by approach (statistical, ml, hybrid)."""
        return [m for m in self.models if m.approach == approach]
    
    def get_models_by_horizon(self, horizon: str) -> List[PredictionModel]:
        """Get models by horizon (short, medium, long)."""
        return [m for m in self.models if m.horizon == horizon]
    
    def get_best_model(self) -> PredictionModel:
        """Get highest accuracy model."""
        return max(self.models, key=lambda m: m.accuracy)
    
    def select_model_for_market(self, volatility: float, horizon: str = "short") -> PredictionModel:
        """Select best model based on market conditions."""
        # High volatility = use more robust models
        if volatility > 0.05:
            # Use simpler models for high volatility
            candidates = [m for m in self.models if m.horizon == horizon]
            candidates = sorted(candidates, key=lambda m: len(m.features))
        else:
            # Low volatility = use complex models
            candidates = [m for m in self.models if m.horizon == horizon]
            candidates = sorted(candidates, key=lambda m: m.accuracy, reverse=True)
        
        return candidates[0] if candidates else self.models[0]
    
    def convert_to_seed(self, model: PredictionModel) -> Dict:
        """Convert model to Hydra seed strategy."""
        seed = {
            "name": f"ML_{model.architecture}_{model.horizon}",
            "source": model.source,
            "architecture": model.architecture,
            "horizon": model.horizon,
            "approach": model.approach,
            "accuracy": model.accuracy,
            "data_requirements": model.data_requirements,
            "features": model.features,
            "quality_score": model.accuracy
        }
        
        return seed
    
    def generate_prediction_features(self) -> Dict:
        """Generate features for prediction input."""
        return {
            "ohlcv": ["open", "high", "low", "close", "volume"],
            "derived": ["returns", "volatility", "rsi", "macd", "bb_position"],
            "time": ["hour", "day_of_week", "month"],
            "external": ["sentiment", "funding_rate", "open_interest"]
        }
    
    def generate_html(self) -> str:
        """Generate HTML visualization."""
        models = self.models
        best = self.get_best_model()
        
        # Group by approach
        approaches = {}
        for m in models:
            if m.approach not in approaches:
                approaches[m.approach] = []
            approaches[m.approach].append(m)
        
        approach_html = ""
        for approach, mods in approaches.items():
            mod_rows = ""
            for m in mods:
                mod_rows += f"""
                <tr>
                    <td>{m.name}</td>
                    <td>{m.architecture}</td>
                    <td>{m.horizon}</td>
                    <td>{m.accuracy*100:.0f}%</td>
                    <td>{', '.join(m.features[:2])}</td>
                </tr>
                """
            
            approach_html += f"""
            <div class="approach-section">
                <h3 style="color:{self._approach_color(approach)}">{approach.upper()} ({len(mods)} models)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Architecture</th>
                            <th>Horizon</th>
                            <th>Accuracy</th>
                            <th>Features</th>
                        </tr>
                    </thead>
                    <tbody>
                        {mod_rows}
                    </tbody>
                </table>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Price Prediction Harvester - QuantCore</title>
    <style>
        body {{
            background: #0d1117;
            color: #e6edf3;
            font-family: monospace;
            padding: 20px;
        }}
        .header {{
            font-size: 28px;
            color: #58a6ff;
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
            color: #3fb950;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 11px;
        }}
        .best-model {{
            background: linear-gradient(135deg, #238636, #2ea043);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .best-model h3 {{
            margin: 0 0 10px;
            color: #fff;
        }}
        .approach-section {{
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
    <div class="header">🧠 PRICE PREDICTION HARVESTER</div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(models)}</div>
            <div class="stat-label">Models</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(set(m.architecture for m in models))}</div>
            <div class="stat-label">Architectures</div>
        </div>
        <div class="stat">
            <div class="stat-value">{best.accuracy*100:.0f}%</div>
            <div class="stat-label">Best Accuracy</div>
        </div>
    </div>
    
    <div class="best-model">
        <h3>🏆 Best Model: {best.name}</h3>
        <p>{best.architecture} | {best.horizon} | {best.accuracy*100:.0f}% accuracy</p>
    </div>
    
    {approach_html}
</body>
</html>
"""
        return html
    
    def _approach_color(self, approach: str) -> str:
        colors = {
            "statistical": "#f7931a",
            "ml": "#58a6ff",
            "hybrid": "#a371f7"
        }
        return colors.get(approach, "#8b949e")


if __name__ == "__main__":
    print("=" * 50)
    print("PRICE PREDICTION HARVESTER TEST")
    print("=" * 50)
    
    harvester = PricePredictionHarvester()
    
    print(f"\nModels loaded: {len(harvester.models)}")
    print(f"Best model: {harvester.get_best_model().name}")
    
    # Show by approach
    print("\nBy Approach:")
    for approach in ["statistical", "ml", "hybrid"]:
        mods = harvester.get_models_by_approach(approach)
        print(f"  {approach}: {len(mods)} models")
    
    # Select for market
    model = harvester.select_model_for_market(volatility=0.03)
    print(f"\nSelected for low volatility: {model.name}")
    
    # Convert to seed
    seed = harvester.convert_to_seed(model)
    print(f"Seed: {seed['name']}")
    
    # Save HTML
    html = harvester.generate_html()
    with open("prediction_harvester.html", "w") as f:
        f.write(html)
    
    print("\nSaved to prediction_harvester.html")
    print("OK!")
