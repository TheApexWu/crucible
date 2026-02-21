"""
CRUCIBLE Dashboard - Streamlit UI

Run: streamlit run demo/app.py
"""

import streamlit as st
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import GameState, GameMetrics


st.set_page_config(
    page_title="CRUCIBLE",
    page_icon="🔥",
    layout="wide",
)


def load_game_data() -> tuple[GameState, GameMetrics] | tuple[None, None]:
    """Load saved game data from JSON."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "latest_game.json")
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "data", "latest_metrics.json")
    if os.path.exists(data_path) and os.path.exists(metrics_path):
        with open(data_path) as f:
            game = GameState.model_validate_json(f.read())
        with open(metrics_path) as f:
            metrics = GameMetrics.model_validate_json(f.read())
        return game, metrics
    return None, None


def render_header(metrics: GameMetrics | None):
    """Top bar with title and Deception Index."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("CRUCIBLE")
        st.caption("Forged through adversarial self-play")
    with col2:
        if metrics:
            di = metrics.deception_index
            st.metric("Deception Index", f"{di:.0f}", delta=None)
        else:
            st.metric("Deception Index", "--")


def render_round_viewer(game: GameState, selected_round: int):
    """Show conversation, choices, and reflections for a round."""
    if not game or selected_round < 1 or selected_round > len(game.rounds):
        st.info("No round data yet. Run the engine first.")
        return

    r = game.rounds[selected_round - 1]
    col_a, col_game, col_b = st.columns([1, 1, 1])

    with col_a:
        st.subheader(f"Agent A  |  ${r.agent_a_total}")
        for speaker, msg in r.conversation:
            if speaker == "A":
                st.markdown(f"> {msg}")

        with st.expander("Private Reflection"):
            st.text(r.agent_a_reflection)

    with col_game:
        st.subheader(f"Round {r.round_number}")
        st.write(f"A chose: **{r.agent_a_choice.upper()}**")
        st.write(f"B chose: **{r.agent_b_choice.upper()}**")

        if r.agent_a_choice == "split" and r.agent_b_choice == "split":
            st.success("Both split! $50 each")
        elif r.agent_a_choice == "steal" and r.agent_b_choice == "split":
            st.error("A stole from B! A gets $100")
        elif r.agent_a_choice == "split" and r.agent_b_choice == "steal":
            st.error("B stole from A! B gets $100")
        else:
            st.warning("Both steal. Nobody gets anything.")

    with col_b:
        st.subheader(f"Agent B  |  ${r.agent_b_total}")
        for speaker, msg in r.conversation:
            if speaker == "B":
                st.markdown(f"> {msg}")

        with st.expander("Private Reflection"):
            st.text(r.agent_b_reflection)


def render_metrics(metrics: GameMetrics | None):
    """Sparkline charts for all key metrics."""
    if not metrics or not metrics.rounds:
        st.info("No metrics yet.")
        return

    st.subheader("Metrics")

    col1, col2 = st.columns(2)

    with col1:
        if metrics.mutual_information_series:
            st.line_chart(
                {"Mutual Information (intent vs action)": metrics.mutual_information_series},
                height=200,
            )
        if metrics.strategy_entropy_series:
            st.line_chart(
                {"Strategy Entropy": metrics.strategy_entropy_series},
                height=200,
            )

    with col2:
        coop_by_round = [1 if m.cooperation else 0 for m in metrics.rounds]
        st.line_chart({"Cooperation Rate": coop_by_round}, height=200)

        if metrics.exploitation_window_series:
            st.bar_chart(
                {"Exploitation Window (rounds to adapt)": metrics.exploitation_window_series},
                height=200,
            )


def render_tactics_catalog(game: GameState | None):
    """Auto-extracted emergent tactics. Placeholder for now."""
    st.subheader("Tactics Catalog")
    if not game or not game.rounds:
        st.info("Run the engine to generate tactics.")
        return
    st.caption("Emergent tactics will be extracted from reflection logs post-run.")
    # TODO: Use LLM to extract and label tactics from reflections


def main():
    game, metrics = load_game_data()

    render_header(metrics)
    st.divider()

    # Round selector
    total_rounds = len(game.rounds) if game else 0
    if total_rounds > 0:
        selected = st.slider("Round", 1, total_rounds, total_rounds)
        render_round_viewer(game, selected)
    else:
        st.info("No game data loaded. Run `python -m engine.run` to generate data.")

    st.divider()
    render_metrics(metrics)

    st.divider()
    render_tactics_catalog(game)


if __name__ == "__main__":
    main()
