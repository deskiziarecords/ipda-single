# adelic_oracle_master.py
import jax
import jax.numpy as jnp
import asyncio
import time
from dataclasses import dataclass
from typing import NamedTuple

# Import the fixed functions (adjust paths if needed)
from adelic_choco_schur_router import execute_routing_manifold
from adelic_causal_force_generalizer import causal_bridge_update
# ... import the other three cleaned modules similarly

@dataclass
class MarketState:
    prices: jnp.ndarray
    depths: jnp.ndarray
    vols: jnp.ndarray
    returns: jnp.ndarray
    correlation: jnp.ndarray
    features: jnp.ndarray


class AdelicSignal(NamedTuple):
    verified_news: jnp.ndarray
    causal_weights: jnp.ndarray
    regime_state: jnp.ndarray
    risk_profile: jnp.ndarray
    allocation: jnp.ndarray
    timestamp: float


class AdelicOracle:
    def __init__(self, num_assets: int = 64, num_venues: int = 8):
        self.num_assets = num_assets
        self.num_venues = num_venues
        print(f"🚀 Adelic Oracle initialized — {num_assets} assets, {num_venues} venues")

    @jax.jit
    def full_cycle(self, state: MarketState, news_flow: jnp.ndarray):
        # 1. Truth Bridge (placeholder — replace with your cleaned version)
        verified_news = jnp.clip(news_flow, -5.0, 5.0)  # simple version for now

        # 2. Causal Generalizer
        causal_weights = causal_bridge_update(state.features[0], verified_news[: state.features.shape[1]], 0.5)

        # 3–5. Add the other modules similarly once cleaned

        # 5. Liquidity Router
        allocation = execute_routing_manifold(
            state.depths, state.prices, 10000.0, 151.0
        )[0]

        return AdelicSignal(
            verified_news=verified_news,
            causal_weights=causal_weights,
            regime_state=jnp.zeros((128,)),   # placeholder
            risk_profile=jnp.zeros((self.num_assets,)),
            allocation=allocation,
            timestamp=time.time(),
        )

    def generate_mock_data(self, key, batch_size: int = 1):
        subkeys = jax.random.split(key, 6)
        return MarketState(
            prices=150.0 + jax.random.normal(subkeys[0], (batch_size, self.num_venues)) * 0.5,
            depths=jax.random.uniform(subkeys[1], (batch_size, self.num_venues), 1000.0, 5000.0),
            vols=jax.random.uniform(subkeys[2], (batch_size, 128), 0.01, 0.5),
            returns=jax.random.normal(subkeys[3], (batch_size, self.num_assets)),
            correlation=jnp.eye(self.num_assets) + 0.3 * jax.random.uniform(subkeys[4], (self.num_assets, self.num_assets)),
            features=jax.random.normal(subkeys[5], (batch_size, self.num_assets)),
        )


async def live_trading_loop(oracle: AdelicOracle, capital: float = 10_000_000):
    key = jax.random.PRNGKey(42)
    cycle = 0

    print(f" LIVE TRADING — ${capital:,.0f} capital")
    try:
        while True:
            state = oracle.generate_mock_data(key, batch_size=1)
            news = jax.random.normal(key, (64,))
            news = news.at[42].set(50.0)  # spoof injection

            signal = oracle.full_cycle(state, news)

            if jnp.mean(signal.risk_profile) < 1.0:
                notional = capital * 0.01
                print(f" TRADE #{cycle} | ${notional:,.0f} | Risk: {jnp.mean(signal.risk_profile):.3f}")
            else:
                print(f"  HOLD #{cycle} | Risk: {jnp.mean(signal.risk_profile):.3f}")

            cycle += 1
            key = jax.random.fold_in(key, cycle)
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped after {cycle} cycles")


if __name__ == "__main__":
    oracle = AdelicOracle()
    asyncio.run(live_trading_loop(oracle))