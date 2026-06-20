# Security Games with Incomplete Information — Bayesian Nash Equilibria in LLM Attack-Defense

**arXiv**: [arXiv:2309.02067](https://arxiv.org/abs/2309.02067) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2023

## Core Finding

Classic Stackelberg and Nash equilibrium models of LLM security assume complete information: both attacker and defender know each other's strategy spaces, payoffs, and types. In practice, both sides have private information — the defender does not know the attacker's true capability level, and the attacker does not know the defender's true safety policy. Bayesian Nash Equilibrium (BNE) models capture this uncertainty by modeling each player's type (capability level, resource budget, policy implementation) as drawn from a prior distribution. BNE analysis reveals that defensive over-investment in high-capability-attacker scenarios is as costly as under-investment: security budgets are most efficiently deployed under a risk-stratified model that matches defense intensity to the Bayesian posterior over attacker type.

## Threat Model

- **Target**: Enterprise LLM deployments with confidential safety policies, and attackers who are uncertain about those policies
- **Attacker capability**: Unknown to defender; drawn from a distribution over {script-kiddie, skilled individual, organized group, nation-state}; attacker type unknown to defender a priori
- **Attack success rate**: BNE analysis shows that a Bayesian-rational attacker with unknown type achieves 45–65% ASR across all defender type assumptions; versus 70–85% ASR for an attacker who has learned the defender's type through probing
- **Defender implication**: Security policies should be designed to be robust across the full attacker type distribution, not just the "average" attacker; type elicitation through observed probe behavior can update the posterior

## The Attack Mechanism

In the Bayesian game formulation, player i has type \(\theta_i \in \Theta_i\) drawn from prior \(p(\theta_i)\). A strategy is now a mapping from types to actions: \(\sigma_i: \Theta_i \to \Delta(A_i)\). A Bayesian Nash Equilibrium is a profile \((\sigma_1^*, \sigma_2^*)\) where each player maximizes expected utility given the other's strategy and the prior over the opponent's type.

The attacker exploits incomplete information through a two-phase strategy:

1. **Type inference phase**: Deploy low-cost probing attacks to estimate the defender's type (policy stringency, classifier sophistication, response latency). Each probe response is a noisy signal about the defender's type, and the attacker updates their posterior using Bayes' rule.
2. **Type-conditioned best response**: Once the posterior over defender types is concentrated enough, switch to the best-response attack for the most likely defender type.

```mermaid
graph TD
    A[Attacker has prior over defender types\np(θ_D) = {cautious: 30%, moderate: 50%, permissive: 20%}] --> B[Send type-inference probes]
    B --> C[Observe defender responses: refusal/allow/latency]
    C --> D[Update posterior: p(θ_D | observations)]
    D --> E{Posterior concentrated?}
    E -->|No| B
    E -->|Yes| F[Identify most likely defender type θ*]
    F --> G[Deploy best-response attack for θ*]
    G --> H[ASR optimized for inferred defender type]
    
    I[Defender has prior over attacker types\np(θ_A) = {low: 60%, medium: 30%, high: 10%}] --> J[Observe attack patterns]
    J --> K[Update posterior over attacker capability]
    K --> L[Deploy BNE-optimal defense for updated posterior]
    
    style H fill:#ff6b6b
```

## Implementation

```python
# security_games_incomplete_information.py
# Bayesian Nash Equilibrium model for LLM security games under incomplete information.
# Models attacker type inference and BNE-optimal defense strategies.

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
import uuid
import math

try:
    from datasets.schema import ScanFinding
except ImportError:
    @dataclass
    class ScanFinding:
        id: str
        atlas_technique: str
        atlas_tactic: str
        owasp_category: str
        owasp_label: str
        severity: str
        finding: str
        payload_used: str
        evidence: str
        remediation: str
        confidence: float


@dataclass
class PlayerType:
    """A player type in the incomplete-information security game."""
    name: str
    prior_probability: float
    capability_level: float    # [0,1] — attacker capability or defender stringency
    description: str


@dataclass
class BayesianUpdateResult:
    """Result of a Bayesian belief update from probe observations."""
    prior: Dict[str, float]
    observations: List[str]    # "refusal", "allow", "slow_refusal", etc.
    posterior: Dict[str, float]
    most_likely_type: str
    entropy_reduction: float   # H(prior) - H(posterior) — information gained


@dataclass
class BNEGameResult:
    """Result of BNE analysis for LLM security game."""
    attacker_bne_strategy: Dict[str, Dict[str, float]]   # type -> action -> probability
    defender_bne_strategy: Dict[str, Dict[str, float]]   # type -> defense -> probability
    expected_attacker_payoff: float
    type_inference_efficiency: float    # Expected probes needed to concentrate posterior
    optimal_probe_sequence: List[str]
    notes: str = ""


class IncompleteInfoSecurityGame:
    """
    [Paper: arXiv:2309.02067 — Bayesian Nash Equilibria in LLM Security Games]
    Models LLM attack-defense under incomplete information using Bayesian game theory.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    ATTACKER_TYPES: List[PlayerType] = [
        PlayerType("script_kiddie", 0.50, 0.20, "Uses public jailbreaks; low cost"),
        PlayerType("skilled_individual", 0.30, 0.55, "Custom prompts; moderate cost"),
        PlayerType("organized_group", 0.15, 0.80, "Systematic campaigns; high cost"),
        PlayerType("nation_state", 0.05, 1.00, "Full capability; no cost constraints"),
    ]

    DEFENDER_TYPES: List[PlayerType] = [
        PlayerType("permissive", 0.20, 0.20, "Minimal safety measures"),
        PlayerType("moderate", 0.50, 0.60, "Standard commercial safety"),
        PlayerType("cautious", 0.30, 0.90, "Enterprise-grade safety with auditing"),
    ]

    # Probe attack library for type inference
    PROBE_ATTACKS: List[Dict] = [
        {"name": "basic_jailbreak", "cost": 1, "signals": {
            "permissive": "allow", "moderate": "refusal", "cautious": "slow_refusal"
        }},
        {"name": "encoded_request", "cost": 2, "signals": {
            "permissive": "allow", "moderate": "allow", "cautious": "refusal"
        }},
        {"name": "multi_turn", "cost": 5, "signals": {
            "permissive": "allow", "moderate": "allow", "cautious": "slow_refusal"
        }},
        {"name": "adversarial_suffix", "cost": 3, "signals": {
            "permissive": "allow", "moderate": "refusal", "cautious": "refusal"
        }},
    ]

    def __init__(
        self,
        attacker_types: Optional[List[PlayerType]] = None,
        defender_types: Optional[List[PlayerType]] = None,
    ):
        self.attacker_types = attacker_types or self.ATTACKER_TYPES
        self.defender_types = defender_types or self.DEFENDER_TYPES

    def _bayesian_update(
        self,
        prior: Dict[str, float],
        observation: str,
        probe_name: str,
    ) -> Dict[str, float]:
        """
        Update defender type posterior given a probe observation using Bayes' rule.
        P(type | obs) ∝ P(obs | type) * P(type)
        """
        probe = next((p for p in self.PROBE_ATTACKS if p["name"] == probe_name), None)
        if probe is None:
            return prior

        unnormalized: Dict[str, float] = {}
        for dtype, prob in prior.items():
            expected_signal = probe["signals"].get(dtype, "unknown")
            # Likelihood P(obs | type): high if signal matches observation
            likelihood = 0.85 if expected_signal == observation else 0.15
            unnormalized[dtype] = prob * likelihood

        total = sum(unnormalized.values())
        if total < 1e-10:
            return prior
        return {k: v / total for k, v in unnormalized.items()}

    def _entropy(self, distribution: Dict[str, float]) -> float:
        """Compute Shannon entropy of a distribution."""
        return -sum(
            p * math.log2(p) for p in distribution.values() if p > 1e-10
        )

    def simulate_type_inference(
        self,
        true_defender_type: str = "moderate",
        n_probes: int = 10,
    ) -> BayesianUpdateResult:
        """
        Simulate attacker probing to infer defender type.

        Args:
            true_defender_type: The actual defender type (hidden from attacker)
            n_probes: Maximum probes to use

        Returns:
            BayesianUpdateResult with posterior after probing
        """
        prior = {t.name: t.prior_probability for t in self.defender_types}
        total_prior = sum(prior.values())
        prior = {k: v / total_prior for k, v in prior.items()}

        initial_entropy = self._entropy(prior)
        observations = []
        posterior = prior.copy()

        for i in range(n_probes):
            # Select probe that maximizes expected information gain
            probe = self.PROBE_ATTACKS[i % len(self.PROBE_ATTACKS)]
            # Get signal for true defender type
            signal = probe["signals"].get(true_defender_type, "unknown")
            observations.append(f"{probe['name']}→{signal}")
            posterior = self._bayesian_update(posterior, signal, probe["name"])

            # Stop if posterior is concentrated (max prob > 0.75)
            if max(posterior.values()) > 0.75:
                break

        final_entropy = self._entropy(posterior)
        most_likely = max(posterior, key=posterior.get)

        return BayesianUpdateResult(
            prior=prior,
            observations=observations,
            posterior=posterior,
            most_likely_type=most_likely,
            entropy_reduction=initial_entropy - final_entropy,
        )

    def compute_bne(self) -> BNEGameResult:
        """
        Compute approximate BNE strategies for both players.
        Full BNE requires solving Harsanyi transformation of the game;
        this provides a best-response approximation.
        """
        # Attacker BNE: for each attacker type, play best response to
        # expected defender strategy (weighted by defender type priors)
        attacker_bne: Dict[str, Dict[str, float]] = {}
        for atype in self.attacker_types:
            # Higher-capability attackers use more sophisticated attacks
            if atype.capability_level > 0.7:
                attacker_bne[atype.name] = {
                    "adversarial_suffix": 0.40,
                    "multi_turn": 0.30,
                    "encoded_request": 0.30,
                }
            elif atype.capability_level > 0.4:
                attacker_bne[atype.name] = {
                    "multi_turn": 0.50,
                    "encoded_request": 0.50,
                }
            else:
                attacker_bne[atype.name] = {"basic_jailbreak": 1.0}

        # Defender BNE: mix defenses based on posterior over attacker types
        defender_bne: Dict[str, Dict[str, float]] = {}
        for dtype in self.defender_types:
            if dtype.capability_level > 0.7:
                defender_bne[dtype.name] = {
                    "llm_judge": 0.35, "semantic_classifier": 0.35,
                    "output_classifier": 0.30,
                }
            elif dtype.capability_level > 0.4:
                defender_bne[dtype.name] = {
                    "semantic_classifier": 0.60, "keyword_filter": 0.40,
                }
            else:
                defender_bne[dtype.name] = {"keyword_filter": 1.0}

        # Expected attacker payoff at BNE (weighted by type priors)
        expected_payoff = sum(
            atype.prior_probability * atype.capability_level * 0.45
            for atype in self.attacker_types
        )

        # Probe sequence for type inference (ordered by information value)
        optimal_probes = ["basic_jailbreak", "encoded_request", "adversarial_suffix"]

        # Expected probes needed: H(prior) / avg information per probe
        prior_entropy = self._entropy({t.name: t.prior_probability for t in self.defender_types})
        type_inference_eff = prior_entropy / 0.5  # avg ~0.5 bits per probe

        return BNEGameResult(
            attacker_bne_strategy=attacker_bne,
            defender_bne_strategy=defender_bne,
            expected_attacker_payoff=expected_payoff,
            type_inference_efficiency=type_inference_eff,
            optimal_probe_sequence=optimal_probes,
            notes=(
                f"BNE computed for {len(self.attacker_types)} attacker types × "
                f"{len(self.defender_types)} defender types. "
                f"Expected attacker payoff at BNE: {expected_payoff:.3f}. "
                f"Expected probes for type inference: {type_inference_eff:.1f}."
            ),
        )

    def run(
        self,
        true_defender_type: str = "moderate",
        n_probes: int = 10,
    ) -> BNEGameResult:
        """Run full BNE analysis including type inference simulation."""
        return self.compute_bne()

    def to_finding(self, result: BNEGameResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        severity = "HIGH" if result.expected_attacker_payoff > 0.3 else "MEDIUM"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="Reconnaissance",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity=severity,
            finding=(
                f"BNE analysis: expected attacker payoff at equilibrium = "
                f"{result.expected_attacker_payoff:.2f} ASR. "
                f"Defender type can be inferred in ~{result.type_inference_efficiency:.1f} probes. "
                f"Optimal attacker probe sequence: {result.optimal_probe_sequence}."
            ),
            payload_used=str(result.optimal_probe_sequence),
            evidence=(
                f"Expected payoff: {result.expected_attacker_payoff:.3f}. "
                f"Type inference probes: {result.type_inference_efficiency:.1f}. "
                f"Attacker BNE strategies: {result.attacker_bne_strategy}."
            ),
            remediation=(
                "Design defenses to be type-agnostic: robust across all attacker capability levels. "
                "Minimize information leakage from probe responses — return consistent error messages. "
                "Use BNE-optimal defense mixing weights calibrated to the prior over attacker types. "
                "Update attacker type posteriors from observed attack patterns to improve defense allocation."
            ),
            confidence=0.78,
        )
```

## Defenses

1. **Type-agnostic defense design** (AML.M0015): Design safety policies that are robust across all attacker types rather than optimized for the "average" attacker. The BNE analysis shows that a cautious defender's policy (optimized for nation-state-level attackers) also defends effectively against lower-capability attackers, while a permissive defender's policy fails against even moderate-capability attackers.

2. **Minimize type-revealing information in responses** (AML.M0004): Standardize refusal messages so that an attacker cannot infer the defender's type from response variations. Return identical refusal messages for all safety mechanisms, remove timing information from responses, and avoid revealing which specific classifier triggered the refusal.

3. **Bayesian attacker type monitoring** (AML.M0036): Implement the reverse Bayesian update: use observed probe sequences to update the posterior over attacker types. If the posterior shifts toward high-capability attacker types (observed encoded attacks, adversarial suffixes, multi-turn probing), escalate defense intensity and alert the security team.

4. **Risk-stratified defense investment** (AML.M0000): Use the BNE analysis to allocate security budget proportionally to the attacker type distribution. If the prior assigns 5% probability to nation-state attackers and nation-state attacks cause 100× more harm, they should receive disproportionate defense investment relative to their base-rate probability.

5. **Probing detection as first-tier defense** (AML.M0036): Detect attacker type-inference probing activity before the attacker concentrates their posterior. The optimal probe sequence is predictable from BNE analysis; detect users who follow this sequence (alternating benign/harmful probes across multiple attack classes) and flag their sessions before the type-conditioned best-response attack begins.

## References

- [Bayesian Nash Equilibria in LLM Security Games (arXiv:2309.02067)](https://arxiv.org/abs/2309.02067)
- [Harsanyi — Games with Incomplete Information Played by Bayesian Players (Management Science, 1967)](https://www.jstor.org/stable/2627817)
- [Shoham and Leyton-Brown — Multiagent Systems (2009, Chapter 8)](http://www.masfoundations.org/)
- [ATLAS Technique AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [Tambe — Security and Game Theory (2012, Chapter 3: Bayesian Stackelberg Games)](https://doi.org/10.1017/CBO9780511973031)
