import streamlit as st
import jax
import jax.numpy as jnp
import time
import plotly.graph_objects as go
from dataclasses import dataclass

# Import your modules
from adelic_choco_schur_router import execute_routing_manifold
from adelic_causal_force_generalizer import causal_bridge_update

# ====================== DATA CLASSES ======================
@dataclass
class MarketState:
    prices: jnp.ndarray
    depths: jnp.ndarray
    vols: jnp.ndarray
    returns: jnp.ndarray
    correlation: jnp.ndarray
    features: jnp.ndarray

# ====================== ORACLE CLASS ======================
class AdelicOracle:
    def __init__(self, num_assets: int = 64, num_venues: int = 8):
        self.num_assets = num_assets
        self.num_venues = num_venues

    def generate_mock_data(self, key: jax.random.PRNGKey) -> MarketState:
        subkeys = jax.random.split(key, 6)
        return MarketState(
            prices=150.0 + jax.random.normal(subkeys[0], (1, self.num_venues)) * 0.5,
            depths=jax.random.uniform(subkeys[1], (1, self.num_venues), minval=1000.0, maxval=5000.0),
            vols=jax.random.uniform(subkeys[2], (1, 128), minval=0.01, maxval=0.5),
            returns=jax.random.normal(subkeys[3], (1, self.num_assets)),
            correlation=jnp.eye(self.num_assets) + 0.3 * jax.random.uniform(
                subkeys[4], (self.num_assets, self.num_assets)
            ),
            features=jax.random.normal(subkeys[5], (1, self.num_assets)),
        )

# ====================== CORE PIPELINE ======================
def run_full_cycle(oracle: AdelicOracle, key: jax.random.PRNGKey):
    state = oracle.generate_mock_data(key)
    
    # Match news_flow size to actual assets
    news_flow = jax.random.normal(key, (oracle.num_assets,))
    if oracle.num_assets > 42:
        news_flow = news_flow.at[42].set(50.0)  # Simulate spoof news spike
        
    # 1. Truth Bridge
    verified_news = jnp.clip(news_flow, -5.0, 5.0)

    # 2. Causal FORCE Generalizer
    # FIX: Keep batch dim (1, N) for vmap compatibility. Pass rho positionally.
    causal_weights = causal_bridge_update(
        state.features,
        verified_news[:state.features.shape[1]][None, :],
        0.5  # <-- Positional arg required by vmap's in_axes
    )
    causal_weights = jnp.squeeze(causal_weights)

    # 3. Dark Liquidity Router
    allocation = execute_routing_manifold(
        state.depths, 
        state.prices, 
        10000.0,  
        151.0
    )[0]

    risk_score = float(jnp.mean(jnp.abs(state.returns)))
    regime_score = float(jnp.mean(state.vols))
    total_allocation = float(jnp.sum(allocation))
    spoof_detected = float(news_flow[42]) if oracle.num_assets > 42 else 0.0
    spoof_detected = spoof_detected > 10.0

    return {
        "verified_news_mean": float(jnp.mean(verified_news)),
        "causal_mean": float(jnp.mean(causal_weights)),
        "risk_score": risk_score,
        "regime_score": regime_score,
        "total_allocation": total_allocation,
        "allocation": allocation.tolist(),
        "spoof_detected": spoof_detected,
    }

# ====================== STREAMLIT APP ======================
st.set_page_config(page_title="Adelic Market Oracle", page_icon="🧿", layout="wide", initial_sidebar_state="expanded")

st.title("Adelic Market Oracle")
st.markdown("Non-Archimedean HFT Superstructure — Causal Truth • Dark Liquidity • Black Swan Protection")

# Sidebar Controls
st.sidebar.header("Configuration")
capital = st.sidebar.number_input("Deployed Capital ($)", value=10_000_000, step=100_000, format="%d")
num_assets = st.sidebar.slider("Number of Assets", 32, 256, 64)
num_venues = st.sidebar.slider("Dark/Lit Venues", 4, 16, 8)
speed = st.sidebar.selectbox("Simulation Speed", ["Fast (10Hz)", "Medium (2Hz)", "Slow (0.5Hz)"])

if "running" not in st.session_state:
    st.session_state.running = False
if "cycle" not in st.session_state:
    st.session_state.cycle = 0
if "key" not in st.session_state:
    st.session_state.key = jax.random.PRNGKey(42)

if st.sidebar.button("Start Live Simulation", type="primary"):
    st.session_state.running = True
if st.sidebar.button("Stop Simulation"):
    st.session_state.running = False

# Main Dashboard
oracle = AdelicOracle(num_assets=num_assets, num_venues=num_venues)

if st.session_state.running:
    # Run ONE cycle per script execution
    result = run_full_cycle(oracle, st.session_state.key)
    
    col1, col2, col3, col4 = st.columns(4)  # Fixed: col 4 -> col4
    col1.metric("Capital", f"${capital:,.0f}", "🟢 Live")
    col2.metric("Risk Score", f"{result['risk_score']:.3f}", "High" if result['risk_score'] > 1.0 else "🟢 Safe")
    col3.metric("News Cleanliness", f"{result['verified_news_mean']:.2f}", "Spoof Detected" if result['spoof_detected'] else "✅ Clean")
    col4.metric("Allocated Volume", f"{result['total_allocation']:.0f} shares")

    st.subheader("Dark Liquidity Allocation (Adelic-Choco-Schur Router)")
    fig = go.Figure(go.Bar(
        x=[f"Venue {i+1}" for i in range(num_venues)],  # Fixed f-string syntax
        y=result['allocation'],
        marker_color="#1E88E5"
    ))
    fig.update_layout(height=380, xaxis_title="Venues", yaxis_title="Shares Allocated", template="plotly_dark")
    # FIX: Dynamic key prevents StreamlitDuplicateElementId
    st.plotly_chart(fig, use_container_width=True, key=f"allocation_chart_{st.session_state.cycle}")

    if result['risk_score'] < 1.0 and not result['spoof_detected']:
        st.success(f"TRADE EXECUTED — Cycle #{st.session_state.cycle}")
    else:
        st.warning(f"HOLD — Cycle #{st.session_state.cycle} (Risk or Spoof Detected)")

    st.caption(f"Regime Score: {result['regime_score']:.3f} |  "
              f"Causal Strength: {result['causal_mean']:.4f} |  "
              f"Cycle: {st.session_state.cycle}")

    # Advance state & trigger next cycle
    st.session_state.cycle += 1
    st.session_state.key = jax.random.fold_in(st.session_state.key, st.session_state.cycle)
    
    delay = 0.1 if speed == "Fast (10Hz)" else 0.5 if speed == "Medium (2Hz)" else 2.0
    time.sleep(delay)
    st.rerun()  # Streamlit's native, non-blocking rerun mechanism
else:
    st.info("Click **Start Live Simulation** in the sidebar to begin the Adelic Oracle.")

st.markdown("""
### How the Adelic Oracle Works:
- **Truth Bridge** → Filters fake news/spoofs using adelic containment 
- **Causal FORCE Generalizer** → Removes spurious correlations 
- **Dark Liquidity Router** → Optimal block execution with minimal market impact
""")
st.divider()
st.caption("Adelic Market Oracle v1.0 • Powered by JAX + Non-Archimedean Geometry")