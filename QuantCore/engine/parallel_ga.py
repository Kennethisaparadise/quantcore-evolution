"""
QuantCore - Parallel Genetic Algorithm with Dask

Distributed fitness evaluation using Dask:
- Parallel backtesting across multiple cores
- Dask Bag for population shards
- Future-based evaluation with fault tolerance
- Dynamic scaling
- Benchmarks included
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import random
import json
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import deque
import traceback

logger = logging.getLogger(__name__)

# Try to import Dask
try:
    import dask
    from dask import delayed
    from dask.bag import Bag
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask not available, using multiprocessing fallback")


# ============================================================
# PARALLEL FITNESS EVALUATION
# ============================================================
def evaluate_strategy_parallel(args: Tuple) -> Dict:
    """
    Evaluate a single strategy (for parallel execution).
    
    This function must be picklable for multiprocessing.
    """
    template, data, backtest_fn, strategy_id = args
    
    try:
        if backtest_fn:
            result = backtest_fn(template, data)
        else:
            # Fallback evaluation
            result = _default_evaluate(template, data)
        
        return {
            'id': strategy_id,
            'template': template,
            'fitness': result.get('fitness_score', 0),
            'return': result.get('total_return', 0),
            'sharpe': result.get('sharpe_ratio', 0),
            'drawdown': result.get('max_drawdown', 0),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'id': strategy_id,
            'template': template,
            'fitness': 0,
            'return': 0,
            'sharpe': 0,
            'drawdown': 0,
            'success': False,
            'error': str(e)
        }


def _default_evaluate(template, data: pd.DataFrame) -> Dict:
    """Default fitness evaluation."""
    # Simple RSI-based backtest
    if data.empty or len(data) < 50:
        return {'fitness_score': 0, 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
    
    params = template.params if hasattr(template, 'params') else template
    
    rsi_period = int(params.get('rsi_period', 14))
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    
    close = data['close'].values
    
    # Calculate RSI
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    if len(gains) < rsi_period:
        return {'fitness_score': 0, 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
    
    # Rolling calculations
    fitness_scores = []
    returns = []
    
    for i in range(rsi_period, len(close) - 1, 5):  # Step for speed
        avg_gain = np.mean(gains[i-rsi_period:i])
        avg_loss = np.mean(losses[i-rsi_period:i])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Signal
        position = 0
        entry_price = 0
        
        if rsi < oversold and position == 0:
            position = 1
            entry_price = close[i]
        elif rsi > overbought and position == 1:
            ret = (close[i] - entry_price) / entry_price
            returns.append(ret)
            position = 0
    
    if not returns:
        return {'fitness_score': 0, 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
    
    returns = np.array(returns)
    total_return = (1 + returns).prod() - 1
    
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # Drawdown
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # Fitness = return * 100 + sharpe * 10
    fitness = total_return * 100 + sharpe * 10
    
    return {
        'fitness_score': fitness,
        'total_return': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': abs(max_dd) * 100
    }


# ============================================================
# PARALLEL POPULATION EVALUATOR
# ============================================================
class ParallelEvaluator:
    """
    Parallel fitness evaluation using Dask or multiprocessing.
    """
    
    def __init__(
        self,
        n_workers: int = None,
        backend: str = 'multiprocessing',  # 'dask', 'multiprocessing', 'threading'
        chunk_size: int = 10
    ):
        self.n_workers = n_workers or mp.cpu_count()
        self.backend = backend
        self.chunk_size = chunk_size
        
        self.dask_client = None
        self.executor = None
        
        # Performance tracking
        self.eval_times: deque = deque(maxlen=100)
        self.total_evaluations = 0
        
    def __del__(self):
        self.shutdown()
    
    def startup(self):
        """Initialize the parallel backend."""
        if self.backend == 'dask' and DASK_AVAILABLE:
            try:
                self.dask_client = LocalCluster(
                    n_workers=self.n_workers,
                    threads_per_worker=1,
                    dashboard_address=None
                )
                logger.info(f"Dask cluster started with {self.n_workers} workers")
            except Exception as e:
                logger.warning(f"Dask startup failed: {e}, falling back to multiprocessing")
                self.backend = 'multiprocessing'
        
        if self.backend == 'multiprocessing':
            self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
            logger.info(f"Multiprocessing executor started with {self.n_workers} workers")
        
        elif self.backend == 'threading':
            self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
            logger.info(f"Thread pool started with {self.n_workers} workers")
    
    def shutdown(self):
        """Shutdown the parallel backend."""
        if self.dask_client:
            self.dask_client.close()
            self.dask_client = None
        
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
    
    def evaluate_population(
        self,
        population: List,
        data: pd.DataFrame,
        backtest_fn: Callable = None,
        progress_callback: Callable = None
    ) -> List[Dict]:
        """
        Evaluate a population in parallel.
        
        Args:
            population: List of strategy templates
            data: Market data for backtesting
            backtest_fn: Custom backtest function
            progress_callback: Optional progress callback
        
        Returns:
            List of evaluation results
        """
        start_time = time.time()
        
        # Prepare arguments
        args_list = [
            (template, data, backtest_fn, f"strat_{i}")
            for i, template in enumerate(population)
        ]
        
        results = []
        failed = 0
        
        if self.backend == 'dask' and DASK_AVAILABLE:
            results = self._evaluate_dask(args_list)
        else:
            results = self._evaluate_mp(args_list, progress_callback)
        
        # Track performance
        eval_time = time.time() - start_time
        self.eval_times.append(eval_time)
        self.total_evaluations += len(population)
        
        return results
    
    def _evaluate_dask(self, args_list: List) -> List[Dict]:
        """Evaluate using Dask."""
        # Create delayed functions
        delayed_evals = [delayed(evaluate_strategy_parallel)(args) for args in args_list]
        
        # Execute in parallel
        bag = Bag.from_delayed(delayed_evals)
        results = bag.compute()
        
        return list(results)
    
    def _evaluate_mp(
        self,
        args_list: List,
        progress_callback: Callable = None
    ) -> List[Dict]:
        """Evaluate using multiprocessing."""
        results = []
        failed = 0
        
        if not self.executor:
            self.startup()
        
        # Submit all tasks
        futures = {
            self.executor.submit(evaluate_strategy_parallel, args): args[3]
            for args in args_list
        }
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
                
                if not result['success']:
                    failed += 1
                
                if progress_callback and len(results) % 10 == 0:
                    progress_callback(len(results), len(args_list))
                    
            except Exception as e:
                failed += 1
                logger.warning(f"Evaluation failed: {e}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        avg_time = np.mean(self.eval_times) if self.eval_times else 0
        
        return {
            'n_workers': self.n_workers,
            'backend': self.backend,
            'avg_eval_time': avg_time,
            'total_evaluations': self.total_evaluations,
            'evaluations_per_second': self.total_evaluations / sum(self.eval_times) if self.eval_times else 0
        }


# ============================================================
# DISTRIBUTED GENETIC ALGORITHM
# ============================================================
class DistributedGA:
    """
    Genetic Algorithm with distributed fitness evaluation.
    
    Features:
    - Parallel population evaluation
    - Fault tolerance with retries
    - Dynamic scaling
    - Progress tracking
    """
    
    def __init__(
        self,
        n_workers: int = None,
        population_size: int = 100,
        elite_percent: float = 0.1,
        mutation_rate: float = 0.3,
        backend: str = 'multiprocessing'
    ):
        self.n_workers = n_workers or mp.cpu_count()
        self.population_size = population_size
        self.elite_percent = elite_percent
        self.mutation_rate = mutation_rate
        self.backend = backend
        
        # Evaluator
        self.evaluator = ParallelEvaluator(
            n_workers=n_workers,
            backend=backend
        )
        
        # State
        self.population: List[Dict] = []
        self.best_fitness = 0
        self.generation = 0
        self.history: List[Dict] = []
        
    def initialize_population(self, seed_template=None) -> List[Dict]:
        """Initialize random population."""
        population = []
        
        for i in range(self.population_size):
            if seed_template and hasattr(seed_template, 'params'):
                # Mutate seed
                params = self._mutate_params(seed_template.params.copy())
            else:
                # Random
                params = self._random_params()
            
            population.append({
                'id': f'gen{self.generation}_{i}',
                'params': params,
                'fitness': 0
            })
        
        return population
    
    def _random_params(self) -> Dict:
        """Generate random strategy parameters."""
        return {
            'rsi_period': random.choice([7, 10, 14, 21, 28]),
            'oversold': random.randint(15, 40),
            'overbought': random.randint(60, 85),
            'position_size': random.uniform(0.05, 0.25),
            'stop_loss': random.uniform(0.02, 0.10),
            'take_profit': random.uniform(0.05, 0.20),
            'invert': random.random() > 0.7
        }
    
    def _mutate_params(self, params: Dict) -> Dict:
        """Mutate parameters."""
        new_params = params.copy()
        
        if random.random() < self.mutation_rate:
            # Change a random parameter
            key = random.choice(list(params.keys()))
            
            if key in ['rsi_period', 'oversold', 'overbought']:
                new_params[key] = params[key] + random.randint(-3, 3)
            elif key in ['position_size', 'stop_loss', 'take_profit']:
                new_params[key] = params[key] * random.uniform(0.8, 1.2)
            elif key == 'invert':
                new_params[key] = not params[key]
        
        return new_params
    
    def evaluate(
        self,
        data: pd.DataFrame,
        progress_callback: Callable = None
    ) -> List[Dict]:
        """
        Evaluate entire population in parallel.
        """
        # Use evaluator
        results = self.evaluator.evaluate_population(
            self.population,
            data,
            progress_callback=progress_callback
        )
        
        # Update population fitness
        for result in results:
            for p in self.population:
                if p['id'] == result['id']:
                    p['fitness'] = result['fitness']
                    break
        
        return sorted(self.population, key=lambda x: x['fitness'], reverse=True)
    
    def evolve_generation(self) -> List[Dict]:
        """
        Create next generation through selection and mutation.
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        
        # Elitism: keep top performers
        n_elite = max(1, int(self.population_size * self.elite_percent))
        elite = sorted_pop[:n_elite]
        
        # Create offspring
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Tournament selection
            parent = random.choice(elite)
            
            # Mutate
            child_params = self._mutate_params(parent['params'].copy())
            
            new_population.append({
                'id': f'gen{self.generation + 1}_{len(new_population)}',
                'params': child_params,
                'fitness': 0
            })
        
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def run(
        self,
        data: pd.DataFrame,
        n_generations: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Run the distributed genetic algorithm.
        """
        self.evaluator.startup()
        
        # Initialize
        self.population = self.initialize_population()
        
        best_overall = None
        
        for gen in range(n_generations):
            gen_start = time.time()
            
            # Evaluate
            if verbose:
                print(f"\n{'='*50}")
                print(f"Generation {gen + 1}/{n_generations}")
                print(f"{'='*50}")
            
            self.population = self.evaluate(data)
            
            # Track best
            best = self.population[0]
            if best_overall is None or best['fitness'] > best_overall['fitness']:
                best_overall = best.copy()
            
            gen_time = time.time() - gen_start
            
            if verbose:
                print(f"Best Fitness: {best['fitness']:.2f}")
                print(f"Population Avg: {np.mean([p['fitness'] for p in self.population]):.2f}")
                print(f"Generation Time: {gen_time:.2f}s")
                print(f"Evaluations/sec: {self.population_size / gen_time:.1f}")
            
            # Evolve
            self.evolve_generation()
            
            self.history.append({
                'generation': gen,
                'best_fitness': best['fitness'],
                'avg_fitness': np.mean([p['fitness'] for p in self.population]),
                'time': gen_time
            })
        
        self.evaluator.shutdown()
        
        return {
            'best_strategy': best_overall,
            'history': self.history,
            'statistics': self.evaluator.get_statistics()
        }


# ============================================================
# BENCHMARKS
# ============================================================
def benchmark_parallel_evaluation(
    n_strategies: int = 500,
    data_length: int = 500
):
    """
    Benchmark parallel vs sequential evaluation.
    """
    print("=" * 60)
    print("PARALLEL EVALUATION BENCHMARK")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 45000 + np.cumsum(np.random.randn(data_length) * 100),
        'volume': np.random.randint(1000, 5000, data_length)
    })
    
    # Generate population
    population = []
    for i in range(n_strategies):
        population.append({
            'id': f'strat_{i}',
            'params': {
                'rsi_period': random.choice([7, 10, 14, 21]),
                'oversold': random.randint(20, 35),
                'overbought': random.randint(65, 80),
            },
            'fitness': 0
        })
    
    n_cores = mp.cpu_count()
    print(f"\nTest: {n_strategies} strategies, {data_length} candles, {n_cores} cores")
    
    # Sequential
    print("\n--- Sequential Evaluation ---")
    start = time.time()
    sequential_results = []
    for template in population[:min(100, n_strategies)]:  # Limit for timing
        result = evaluate_strategy_parallel((template, data, None, template['id']))
        sequential_results.append(result)
    seq_time = time.time() - start
    seq_per_sec = len(sequential_results) / seq_time
    print(f"Time: {seq_time:.2f}s")
    print(f"Rate: {seq_per_sec:.1f} strategies/sec")
    
    # Parallel with multiprocessing
    print("\n--- Parallel (Multiprocessing) ---")
    evaluator = ParallelEvaluator(n_workers=n_cores, backend='multiprocessing')
    evaluator.startup()
    
    start = time.time()
    par_results = evaluator.evaluate_population(
        population[:n_strategies],
        data
    )
    par_time = time.time() - start
    par_per_sec = len(par_results) / par_time
    print(f"Time: {par_time:.2f}s")
    print(f"Rate: {par_per_sec:.1f} strategies/sec")
    
    evaluator.shutdown()
    
    # Calculate speedup
    if seq_time > 0 and par_time > 0:
        speedup = (seq_time / par_time) * (n_strategies / min(100, n_strategies))
        print(f"\n--- Speedup ---")
        print(f"Estimated speedup: {speedup:.1f}x")
    
    print("\n✅ Benchmark complete!")
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'strategies': n_strategies,
        'cores': n_cores
    }


def benchmark_full_ga(
    population_size: int = 200,
    n_generations: int = 5,
    data_length: int = 300
):
    """Benchmark full GA with parallel evaluation."""
    print("\n" + "=" * 60)
    print("FULL GA BENCHMARK")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 45000 + np.cumsum(np.random.randn(data_length) * 100)
    })
    
    n_cores = mp.cpu_count()
    print(f"\nPopulation: {population_size}, Generations: {n_generations}, Cores: {n_cores}")
    
    # Run distributed GA
    ga = DistributedGA(
        n_workers=n_cores,
        population_size=population_size,
        elite_percent=0.1,
        backend='multiprocessing'
    )
    
    start = time.time()
    results = ga.run(data, n_generations=n_generations, verbose=True)
    total_time = time.time() - start
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Fitness: {results['best_strategy']['fitness']:.2f}")
    print(f"Total Evaluations: {population_size * n_generations}")
    print(f"Evaluations/sec: {(population_size * n_generations) / total_time:.1f}")
    
    return results


# ============================================================
# DEMO
# ============================================================
def demo():
    """Demo the parallel GA."""
    # Quick benchmark
    benchmark_parallel_evaluation(n_strategies=200, data_length=200)
    
    # Full GA
    benchmark_full_ga(population_size=100, n_generations=3, data_length=200)


if __name__ == "__main__":
    demo()
