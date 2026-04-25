"""Post-simulation agent interview system."""

from __future__ import annotations

from typing import Protocol

from agent_urban_planning.llm.clients import LLMClient, _build_persona
from agent_urban_planning.core.results import AgentResult, SimulationResults


class Interview:
    """Post-simulation conversational interview with a specific agent.

    Constructs an LLM prompt grounded in the agent's persona and
    actual simulation outcome, supporting multi-turn follow-ups.
    """

    def __init__(
        self,
        agent_result: AgentResult,
        simulation_results: SimulationResults,
        client: LLMClient,
    ):
        self.agent_result = agent_result
        self.simulation_results = simulation_results
        self.client = client
        self.history: list[tuple[str, str]] = []  # (question, answer) pairs
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        r = self.agent_result
        demo = r.demographics

        # Agent persona
        parts = [
            f"You are a {demo['age_head']}-year-old head of a household of {demo['household_size']}.",
            f"Your monthly household income is S${demo['income']:.0f}.",
            f"You work in the {demo['job_location']} area.",
        ]
        if demo.get("has_children"):
            parts.append("You have school-age children.")
        if demo.get("has_elderly"):
            parts.append("You have elderly family members who need regular healthcare.")
        if demo.get("car_owner"):
            parts.append("You own a car.")
        else:
            parts.append("You do not own a car and rely on public transport.")

        persona = " ".join(parts)

        # Simulation outcome
        outcome_parts = [
            f"\nIn a recent urban development scenario, you chose to live in the {r.zone_choice} zone.",
            f"Your monthly rent there is S${r.equilibrium_price:.0f}.",
            f"Your commute to work is {r.commute_minutes:.0f} minutes.",
        ]

        # Facilities at chosen zone
        metrics = self.simulation_results.metrics
        zone_prices = metrics.zone_prices

        # Other zones for counterfactual context
        other_zones = []
        for zone_name, utility in r.zone_utilities.items():
            if zone_name != r.zone_choice:
                price = zone_prices.get(zone_name, 0)
                other_zones.append(
                    f"  - {zone_name}: utility {utility:.3f}, price S${price:.0f}/month"
                )

        if other_zones:
            outcome_parts.append("\nYour other options were:")
            outcome_parts.extend(other_zones)

        outcome_parts.append(
            f"\nYour overall satisfaction (utility) with your choice is {r.realized_utility:.3f}."
        )

        if r.utility_vs_baseline is not None:
            if r.utility_vs_baseline > 0:
                outcome_parts.append(
                    f"This is an improvement of {r.utility_vs_baseline:.3f} compared to the baseline scenario."
                )
            elif r.utility_vs_baseline < 0:
                outcome_parts.append(
                    f"This is a decrease of {abs(r.utility_vs_baseline):.3f} compared to the baseline scenario."
                )

        outcome = "\n".join(outcome_parts)

        # Preference context
        prefs = r.preferences
        pref_text = (
            f"\nYour priorities: housing affordability ({prefs['alpha']:.0%}), "
            f"commute ({prefs['beta']:.0%}), "
            f"services/facilities ({prefs['gamma']:.0%}), "
            f"amenities ({prefs['delta']:.0%})."
        )

        return (
            f"{persona}\n{outcome}\n{pref_text}\n\n"
            "Respond in first person as this person. Be specific and grounded "
            "in your actual situation — your income, family, commute, and the "
            "facilities available to you. Keep responses conversational and "
            "concise (2-4 sentences unless asked for detail)."
        )

    def ask(self, question: str) -> str:
        """Ask the agent a question and get an in-character response."""
        # Build the full prompt with history
        messages = [self._system_prompt, ""]
        for q, a in self.history:
            messages.append(f"Interviewer: {q}")
            messages.append(f"You: {a}")
            messages.append("")

        messages.append(f"Interviewer: {question}")
        messages.append("You:")

        prompt = "\n".join(messages)
        response = self.client.complete(prompt, system=self._system_prompt)

        # Store in history
        self.history.append((question, response))
        return response

    @property
    def system_prompt(self) -> str:
        """Access the system prompt for inspection/testing."""
        return self._system_prompt
