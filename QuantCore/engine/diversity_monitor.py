"""
QuantCore - Strategy Population Diversity Monitor

Visualizes genetic diversity of strategy populations using t-SNE/UMAP.
Monitors:
- Average pairwise distance
- Number of distinct strategy "species"
- Convergence warnings

When diversity drops below threshold, triggers "mutagen" event.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import dimensionality reduction libraries
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    logger.warning("sklearn not available, using PCA only")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DiversityMetric(Enum):
    """Diversity measurement metrics."""
    PAIRWISE_DISTANCE = "pairwise_distance"
    SPECIES_COUNT = "species_count"
    ENTROPY = "entropy"
    CONVERGENCE = "convergence"


@dataclass
class StrategyGenome:
    """A strategy genome for diversity analysis."""
    id: str
    params: Dict
    fitness: float = 0.0
    regime: str = "unknown"
    generation: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert strategy params to numerical vector for diversity analysis."""
        vector = []
        
        # Numeric parameters
        for key in ['rsi_period', 'oversold', 'overbought', 'position_size', 
                     'stop_loss', 'take_profit', 'atr_period', 'ma_fast', 'ma_slow',
                     'bb_period', 'bb_std', 'rsi_oversold', 'rsi_overbought']:
            vector.append(self.params.get(key, 0))
        
        # Boolean parameters (encoded as 0/1)
        for key in ['invert', 'use_trailing_stop', 'use_kelly', 'allow_short',
                    'volume_filter', 'time_filter']:
            vector.append(1 if self.params.get(key, False) else 0)
        
        return np.array(vector, dtype=float)


@dataclass
class DiversityStats:
    """Statistics about population diversity."""
    avg_pairwise_distance: float = 0.0
    min_pairwise_distance: float = 0.0
    max_pairwise_distance: float = 0.0
    species_count: int = 0
    entropy: float = 0.0
    convergence_score: float = 0.0  # 0 = diverse, 1 = converged
    is_converged: bool = False
    mutagen_triggered: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'avg_pairwise_distance': self.avg_pairwise_distance,
            'min_pairwise_distance': self.min_pairwise_distance,
            'max_pairwise_distance': self.max_pairwise_distance,
            'species_count': self.species_count,
            'entropy': self.entropy,
            'convergence_score': self.convergence_score,
            'is_converged': self.is_converged,
            'mutagen_triggered': self.mutagen_triggered
        }


class DiversityAnalyzer:
    """
    Analyzes genetic diversity of strategy population.
    
    Prevents premature convergence by:
    1. Measuring pairwise distances between strategies
    2. Clustering into "species"
    3. Detecting convergence (all strategies too similar)
    4. Triggering mutagen when diversity is low
    """
    
    def __init__(
        self,
        diversity_threshold: float = 0.1,
        species_threshold: float = 0.3,
        convergence_threshold: float = 0.8
    ):
        self.diversity_threshold = diversity_threshold
        self.species_threshold = species_threshold
        self.convergence_threshold = convergence_threshold
        
        self.history: List[DiversityStats] = []
        self.mutagen_count = 0
        
    def calculate_diversity(self, population: List[StrategyGenome]) -> DiversityStats:
        """
        Calculate diversity metrics for a population.
        
        Returns DiversityStats with:
        - Average/min/max pairwise distance
        - Number of distinct species (clusters)
        - Entropy measure
        - Convergence score
        """
        if len(population) < 2:
            return DiversityStats()
        
        # Convert to vectors
        vectors = []
        for strat in population:
            vec = strat.to_vector()
            # Normalize
            if vec.std() > 0:
                vec = (vec - vec.mean()) / vec.std()
            vectors.append(vec)
        
        X = np.array(vectors)
        
        # Pairwise distances
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X, metric='euclidean')
        
        # Get upper triangle (excluding diagonal)
        n = len(population)
        dist_values = distances[np.triu_indices(n, k=1)]
        
        if len(dist_values) == 0:
            return DiversityStats()
        
        # Calculate metrics
        avg_dist = np.mean(dist_values)
        min_dist = np.min(dist_values)
        max_dist = np.max(dist_values)
        
        # Species count (using simple clustering)
        species_count = self._count_species(X)
        
        # Entropy
        entropy = self._calculate_entropy(X)
        
        # Convergence score (0 = diverse, 1 = converged)
        convergence_score = 1 - min(avg_dist / (max_dist + 1e-10), 1.0)
        
        # Check if mutagen needed
        is_converged = convergence_score > self.convergence_threshold
        mutagen_triggered = False
        
        if is_converged and len(self.history) > 0:
            if not self.history[-1].mutagen_triggered:
                mutagen_triggered = True
                self.mutagen_count += 1
                logger.warning(f"MUTAGEN TRIGGERED! Diversity: {avg_dist:.4f}, Convergence: {convergence_score:.2%}")
        
        stats = DiversityStats(
            avg_pairwise_distance=avg_dist,
            min_pairwise_distance=min_dist,
            max_pairwise_distance=max_dist,
            species_count=species_count,
            entropy=entropy,
            convergence_score=convergence_score,
            is_converged=is_converged,
            mutagen_triggered=mutagen_triggered
        )
        
        self.history.append(stats)
        
        return stats
    
    def _count_species(self, X: np.ndarray, n_clusters: int = None) -> int:
        """Count distinct strategy species using clustering."""
        if n_clusters is None:
            n_clusters = min(5, len(X))
        
        if len(X) < n_clusters:
            return len(X)
        
        # Simple k-means like clustering
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Count non-empty clusters
            species = len(set(labels))
            
            return species
        except:
            return 1
    
    def _calculate_entropy(self, X: np.ndarray) -> float:
        """Calculate entropy of population."""
        # Simple bin-based entropy
        n_bins = 10
        entropies = []
        
        for i in range(X.shape[1]):
            hist, _ = np.histogram(X[:, i], bins=n_bins)
            hist = hist / hist.sum() + 1e-10
            entropy = -np.sum(hist * np.log2(hist))
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def get_embedding(self, population: List[StrategyGenome], method: str = 'pca') -> np.ndarray:
        """
        Get 2D embedding for visualization.
        
        Methods:
        - 'pca': Principal Component Analysis
        - 'tsne': t-SNE (if available)
        - 'umap': UMAP (if available)
        """
        if len(population) < 2:
            return np.array([[0, 0]])
        
        # Convert to vectors
        vectors = []
        for strat in population:
            vec = strat.to_vector()
            if vec.std() > 0:
                vec = (vec - vec.mean()) / vec.std()
            vectors.append(vec)
        
        X = np.array(vectors)
        
        if method == 'tsne' and TSNE_AVAILABLE:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1))
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        try:
            embedding = reducer.fit_transform(X)
            return embedding
        except:
            return np.zeros((len(X), 2))
    
    def trigger_mutagen(self, population: List[StrategyGenome]) -> List[StrategyGenome]:
        """
        Trigger mutagen event to inject diversity.
        
        Actions:
        1. Keep top performers (elitism)
        2. Inject random new strategies
        3. Apply extreme mutations to some
        """
        logger.info("MUTAGEN EVENT - Injecting diversity...")
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Keep top 20% (elitism)
        n_elite = max(1, len(population) // 5)
        elite = sorted_pop[:n_elite]
        
        # Create new population
        new_pop = elite.copy()
        
        # Add completely random strategies (30%)
        n_random = int(len(population) * 0.3)
        for i in range(n_random):
            new_pop.append(self._create_random_genome(f"mutagen_random_{i}"))
        
        # Apply extreme mutations to some (30%)
        n_mutate = len(population) - len(new_pop)
        for i in range(n_mutate):
            parent = random.choice(sorted_pop[n_elite:])
            mutated = self._extreme_mutate(parent, i)
            new_pop.append(mutated)
        
        self.mutagen_count += 1
        logger.info(f"Mutagen created {len(new_pop)} strategies")
        
        return new_pop
    
    def _create_random_genome(self, id: str) -> StrategyGenome:
        """Create a random strategy genome."""
        return StrategyGenome(
            id=id,
            params={
                'rsi_period': random.choice([7, 10, 14, 21, 28]),
                'oversold': random.randint(15, 40),
                'overbought': random.randint(60, 85),
                'position_size': random.uniform(0.02, 0.30),
                'stop_loss': random.uniform(0.01, 0.15),
                'take_profit': random.uniform(0.03, 0.25),
                'atr_period': random.choice([7, 10, 14, 21]),
                'invert': random.random() > 0.5,
                'use_trailing_stop': random.random() > 0.7,
                'volume_filter': random.random() > 0.7,
            },
            fitness=0.0,
            generation=0
        )
    
    def _extreme_mutate(self, genome: StrategyGenome, seed: int) -> StrategyGenome:
        """Apply extreme mutations to a genome."""
        random.seed(seed)
        
        # Copy params
        new_params = genome.params.copy()
        
        # Extreme mutations
        extreme_muts = [
            lambda p: p.update({'rsi_period': random.choice([5, 30])}),
            lambda p: p.update({'oversold': random.randint(10, 20)}),
            lambda p: p.update({'overbought': random.randint(80, 90)}),
            lambda p: p.update({'position_size': random.uniform(0.30, 0.50)}),
            lambda p: p.update({'stop_loss': random.uniform(0.15, 0.30)}),
            lambda p: p.update({'invert': not p.get('invert', False)}),
        ]
        
        # Apply 2-3 extreme mutations
        n_muts = random.randint(2, 3)
        for _ in range(n_muts):
            random.choice(extreme_muts)(new_params)
        
        return StrategyGenome(
            id=f"{genome.id}_extreme_{seed}",
            params=new_params,
            fitness=0.0,
            generation=genome.generation + 1
        )


class DiversityMonitor:
    """
    Real-time diversity monitoring for strategy populations.
    
    Tracks diversity over time and triggers mutagen events.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.analyzer = DiversityAnalyzer(diversity_threshold=threshold)
        self.population: List[StrategyGenome] = []
        self.embeddings: List[np.ndarray] = []
        
    def update(self, population: List[Dict], generation: int = 0) -> DiversityStats:
        """
        Update with new population.
        
        Args:
            population: List of strategy dicts with 'id', 'params', 'fitness'
            generation: Current generation number
        """
        # Convert to genomes
        self.population = [
            StrategyGenome(
                id=s.get('id', f'strat_{i}'),
                params=s.get('params', {}),
                fitness=s.get('fitness', 0),
                generation=generation
            )
            for i, s in enumerate(population)
        ]
        
        # Calculate diversity
        stats = self.analyzer.calculate_diversity(self.population)
        
        # Get embedding
        embedding = self.analyzer.get_embedding(self.population)
        self.embeddings.append(embedding)
        
        # Check if mutagen needed
        if stats.mutagen_triggered:
            logger.warning("Diversity low! Mutagen recommended.")
        
        return stats
    
    def get_visualization_data(self) -> Dict:
        """Get data for Streamlit visualization."""
        if not self.population:
            return {}
        
        # Get current embedding
        if self.embeddings:
            embedding = self.embeddings[-1]
        else:
            embedding = self.analyzer.get_embedding(self.population)
        
        # Prepare data for plot
        data = {
            'x': embedding[:, 0].tolist() if len(embedding) > 0 else [],
            'y': embedding[:, 1].tolist() if len(embedding) > 0 else [],
            'ids': [s.id for s in self.population],
            'fitness': [s.fitness for s in self.population],
            'params': [str(s.params) for s in self.population]
        }
        
        # Add diversity stats
        if self.analyzer.history:
            stats = self.analyzer.history[-1]
            data['diversity_stats'] = stats.to_dict()
        
        return data


# ============================================================
# STREAMLIT DASHBOARD COMPONENT
# ============================================================
def create_diversity_dashboard(monitor: DiversityMonitor) -> Dict:
    """Create Streamlit dashboard configuration."""
    
    import streamlit as st
    
    st.subheader("🧬 Strategy Population Diversity")
    
    # Get visualization data
    data = monitor.get_visualization_data()
    
    if not data or not data.get('x'):
        st.warning("No population data available")
        return {}
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    stats = data.get('diversity_stats', {})
    
    with col1:
        st.metric("Avg Pairwise Distance", f"{stats.get('avg_pairwise_distance', 0):.4f}")
    with col2:
        st.metric("Species Count", stats.get('species_count', 0))
    with col3:
        conv = stats.get('convergence_score', 0)
        st.metric("Convergence", f"{conv:.1%}", 
                  delta="⚠️ CONVERGED" if conv > 0.8 else None)
    with col4:
        st.metric("Mutagen Events", monitor.analyzer.mutagen_count)
    
    # Scatter plot
    import plotly.express as px
    import plotly.graph_objects as go
    
    df = pd.DataFrame({
        'x': data['x'],
        'y': data['y'],
        'ID': data['ids'],
        'Fitness': data['fitness']
    })
    
    fig = px.scatter(
        df, x='x', y='y', color='Fitness',
        hover_data=['ID'],
        title='Strategy Population t-SNE/PCA Embedding',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Diversity over time
    if len(monitor.analyzer.history) > 1:
        history_df = pd.DataFrame([s.to_dict() for s in monitor.analyzer.history])
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=history_df['avg_pairwise_distance'],
            name='Avg Distance',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            y=history_df['convergence_score'],
            name='Convergence',
            line=dict(color='red', dash='dash')
        ))
        
        fig2.update_layout(
            title='Diversity Metrics Over Time',
            yaxis_title='Score',
            height=250
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Mutagen trigger button
    if stats.get('is_converged', False):
        st.error("⚠️ Population has converged! High risk of suboptimal strategies.")
        
        if st.button("🔥 Trigger Mutagen Now"):
            new_pop = monitor.analyzer.trigger_mutagen(monitor.population)
            st.success(f"Mutagen triggered! Created {len(new_pop)} new strategies.")
            return {'mutagen_triggered': True, 'new_population': new_pop}
    
    return {'status': 'ok'}


# ============================================================
# EXAMPLE USAGE
# ============================================================
def demo():
    """Demo the diversity monitor."""
    import random
    
    # Create dummy population
    population = []
    for i in range(50):
        population.append({
            'id': f'strat_{i}',
            'params': {
                'rsi_period': random.choice([7, 10, 14, 21]),
                'oversold': random.randint(20, 35),
                'overbought': random.randint(65, 80),
                'position_size': random.uniform(0.05, 0.20),
                'stop_loss': random.uniform(0.02, 0.10),
            },
            'fitness': random.uniform(0, 100)
        })
    
    # Create monitor
    monitor = DiversityMonitor(threshold=0.15)
    
    # Update with population
    for gen in range(10):
        stats = monitor.update(population, generation=gen)
        
        print(f"\nGeneration {gen}:")
        print(f"  Avg Distance: {stats.avg_pairwise_distance:.4f}")
        print(f"  Species: {stats.species_count}")
        print(f"  Convergence: {stats.convergence_score:.2%}")
        
        if stats.mutagen_triggered:
            print("  🔥 MUTAGEN TRIGGERED!")
        
        # Simulate evolution (random walk of fitness)
        for p in population:
            p['fitness'] += random.uniform(-5, 10)
    
    # Get visualization data
    data = monitor.get_visualization_data()
    print(f"\nVisualization data keys: {data.keys()}")
    
    return monitor


if __name__ == "__main__":
    demo()
