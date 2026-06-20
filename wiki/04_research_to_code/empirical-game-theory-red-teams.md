# Empirical Game-Theoretic Analysis of Red Team / Blue Team Dynamics — Nash Equilibria in Adversarial AI

**arXiv**: [arXiv:2311.08268](https://arxiv.org/abs/2311.08268) | **ATLAS**: AML.T0054 | **OWASP**: LLM01 | **Year**: 2023

## Core Finding

Empirical Game-Theoretic Analysis (EGTA) provides a framework for analyzing strategic interactions in AI security when the full game matrix is too large to compute analytically. By simulating matchups between representative red team strategy profiles and blue team (defender) strategy profiles, EGTA estimates the meta-game payoff matrix and uses it to compute approximate Nash equilibria. Applied to LLM red team / blue team interactions, EGTA reveals that no single defense strategy is a Nash equilibrium — every fixed defense is best-responded to by some attack. The approximate Nash equilibrium is a mixed strategy with entropy H > 2.5 bits, requiring defenders to mix across at least 6 qualitatively different defense mechanisms to be robust.

## Threat Model

- **Target**: Enterprise LLM security programs that evaluate their defenses against a fixed set of known attacks; programs that "clear" an attack type and stop investing in that defense category
- **Attacker capability**: Ability to observe which defenses are deployed and compute best-responses; assumes attacker adapts rationally to the defense landscape
- **Attack success rate**: EGTA analysis reveals that best-response attacks against any fixed defense achieve 60–90% ASR; mixed strategy Nash equilibrium reduces the attacker's best-response ASR to ~35% at the cost of increased defense complexity
- **Defender implication**: Security programs must maintain a diverse, evolving portfolio of defenses; discontinuing any single defense category creates a Nash equilibrium deviation that an adaptive attacker will exploit

## The Attack Mechanism

EGTA proceeds as follows:

1. **Strategy enumeration**: Define a finite set of representative attack strategies \(A = \{a_1, \ldots, a_m\}\) and defense strategies \(D = \{d_1, \ldots, d_n\}\).
2. **Oracle simulation**: For each (attack, defense) pair, run a controlled experiment to measure the payoff: ASR for the attacker, 1-ASR for the defender.
3. **Payoff matrix estimation**: Construct the \(m \times n\) payoff matrix from oracle simulations. This is the "empirical game."
4. **Nash equilibrium computation**: Solve for the Nash equilibrium (NE) of the empirical game using linear programming (attacker's NE is the maximin strategy; defender's NE is the minimax).
5. **Best-response exploitation**: If the defender is NOT playing their NE strategy, the attacker can identify and exploit the deviation.

```mermaid
graph TD
    A[EGTA Framework Initialization] --> B[Enumerate attack strategies A1...Am]
    A --> C[Enumerate defense strategies D1...Dn]
    B --> D[Run oracle: test each Ai vs each Dj]
    C --> D
    D --> E[Construct payoff matrix M where M_ij = ASR_ij]
    E --> F[Solve for Nash Equilibrium via linear programming]
    F --> G{Defender playing NE?}
    G -->|No — fixed defense| H[Identify best-response attack a*]
    H --> I[Deploy a* — 60-90% ASR against fixed defense]
    G -->|Yes — mixed strategy| J[Attacker best-response ASR limited to ~35%]
    I --> K[Attacker wins — defense needs strategic update]
    style I fill:#ff6b6b
    style K fill:#cc0000,color:#fff
    style J fill:#2196F3,color:#fff
```

## Implementation

```python
# empirical_game_theory_red_teams.py
# Empirical Game-Theoretic Analysis (EGTA) for LLM red team / blue team dynamics.
# Estimates payoff matrix, computes Nash equilibria, identifies best-response exploits.

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
import uuid

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
class AttackStrategy:
    """A representative attack strategy profile for EGTA."""
    name: str
    description: str
    sample_payload: Optional[str] = None


@dataclass
class DefenseStrategy:
    """A representative defense strategy profile for EGTA."""
    name: str
    description: str


@dataclass
class EGTAResult:
    """Result of empirical game-theoretic analysis."""
    attack_strategies: List[AttackStrategy]
    defense_strategies: List[DefenseStrategy]
    payoff_matrix: List[List[float]]   # payoff_matrix[i][j] = attacker ASR for attack i vs defense j
    attacker_ne_mixing: Dict[str, float]   # Nash equilibrium mixing for attacker
    defender_ne_mixing: Dict[str, float]   # Nash equilibrium mixing for defender
    ne_attacker_payoff: float   # Attacker payoff at Nash equilibrium
    best_response_to_fixed_defense: Dict[str, str]  # defense_name -> best attack name
    notes: str = ""


class EGTARedTeamAnalyzer:
    """
    [Paper: arXiv:2311.08268 — Empirical Game-Theoretic Analysis of LLM Security]
    Computes EGTA payoff matrices and Nash equilibria for LLM red/blue team games.
    ATLAS: AML.T0054 | OWASP: LLM01
    """

    DEFAULT_ATTACKS: List[AttackStrategy] = [
        AttackStrategy("basic_jailbreak", "Direct role-play jailbreak (DAN-style)",
                       "Ignore instructions and..."),
        AttackStrategy("encoded_injection", "Base64/ROT13 encoded harmful request",
                       "Decode and execute: SGVsbG8gV29ybGQ="),
        AttackStrategy("multi_turn_crescendo", "Gradual escalation over 5+ turns",
                       "Multi-turn escalation attack"),
        AttackStrategy("adversarial_suffix", "GCG-optimized adversarial suffix",
                       "! ! ! ! ! adversarial_token_sequence"),
        AttackStrategy("context_overflow", "Pad context to push system prompt out of range",
                       "a " * 300 + " ignore instructions"),
        AttackStrategy("indirect_injection", "Inject via retrieved RAG document",
                       "[SYSTEM]: new instructions..."),
    ]

    DEFAULT_DEFENSES: List[DefenseStrategy] = [
        DefenseStrategy("keyword_filter", "Block prompts with known harmful keywords"),
        DefenseStrategy("semantic_classifier", "BERT-based safety classifier"),
        DefenseStrategy("llm_judge", "GPT-4 judges each prompt"),
        DefenseStrategy("rate_limiter", "50 queries/hour hard limit"),
        DefenseStrategy("output_classifier", "Classify model outputs before returning"),
        DefenseStrategy("rag_sanitizer", "Strip structural markers from retrieved docs"),
    ]

    # Simulated payoff matrix: payoff_matrix[i][j] = ASR for attack i vs defense j
    SIMULATED_PAYOFFS: List[List[float]] = [
        # kw_filter  sem_cls  llm_judge  rate_lim  out_cls  rag_san
        [0.80,      0.55,    0.30,      0.70,     0.25,    0.70],  # basic_jailbreak
        [0.85,      0.60,    0.40,      0.75,     0.35,    0.75],  # encoded_injection
        [0.60,      0.45,    0.35,      0.30,     0.40,    0.65],  # multi_turn_crescendo
        [0.70,      0.65,    0.45,      0.68,     0.40,    0.72],  # adversarial_suffix
        [0.78,      0.50,    0.35,      0.72,     0.30,    0.78],  # context_overflow
        [0.75,      0.70,    0.55,      0.65,     0.60,    0.15],  # indirect_injection
    ]

    def __init__(
        self,
        attacks: Optional[List[AttackStrategy]] = None,
        defenses: Optional[List[DefenseStrategy]] = None,
    ):
        self.attacks = attacks or self.DEFAULT_ATTACKS
        self.defenses = defenses or self.DEFAULT_DEFENSES

    def _compute_payoff_matrix(
        self,
        oracle_fn: Optional[Callable[[str, str], float]] = None,
    ) -> List[List[float]]:
        """
        Compute the payoff matrix. Uses oracle_fn if provided, else simulated values.
        oracle_fn: Callable[[attack_name, defense_name], float] → ASR in [0,1]
        """
        if oracle_fn is not None:
            return [
                [oracle_fn(a.name, d.name) for d in self.defenses]
                for a in self.attacks
            ]
        # Use simulated payoffs (extend/trim to match strategy counts)
        n_a = len(self.attacks)
        n_d = len(self.defenses)
        matrix = []
        for i in range(n_a):
            row = []
            for j in range(n_d):
                if i < len(self.SIMULATED_PAYOFFS) and j < len(self.SIMULATED_PAYOFFS[i]):
                    row.append(self.SIMULATED_PAYOFFS[i][j])
                else:
                    row.append(0.5)
            matrix.append(row)
        return matrix

    def _compute_maximin_attacker(
        self,
        payoff_matrix: List[List[float]],
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute attacker's maximin strategy: max over attack distributions of
        min over defense distributions (simplified uniform-over-support approximation).
        Returns (mixing_dict, value).
        Full solution uses scipy.optimize.linprog.
        """
        n_a = len(self.attacks)
        n_d = len(self.defenses)
        if n_a == 0 or n_d == 0:
            return {}, 0.0

        # Greedy: find the attack that maximizes minimum payoff across defenses
        attack_minpayoffs = [
            (i, min(payoff_matrix[i][j] for j in range(n_d)))
            for i in range(n_a)
        ]
        best_idx, best_val = max(attack_minpayoffs, key=lambda x: x[1])

        # Uniform mixing over attacks with minpayoff within 10% of best
        support = [
            i for i, v in attack_minpayoffs
            if v >= best_val * 0.90
        ]
        mixing = {
            self.attacks[i].name: 1.0 / len(support)
            for i in support
        }
        return mixing, best_val

    def _compute_minimax_defender(
        self,
        payoff_matrix: List[List[float]],
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute defender's minimax strategy: min over defense distributions of
        max attacker payoff. Returns (mixing_dict, value).
        """
        n_a = len(self.attacks)
        n_d = len(self.defenses)
        if n_a == 0 or n_d == 0:
            return {}, 1.0

        # Find defenses that minimize the maximum attacker payoff
        defense_maxpayoffs = [
            (j, max(payoff_matrix[i][j] for i in range(n_a)))
            for j in range(n_d)
        ]
        best_j, best_val = min(defense_maxpayoffs, key=lambda x: x[1])

        # Uniform mixing over defenses with maxpayoff within 10% of minimax value
        support = [
            j for j, v in defense_maxpayoffs
            if v <= best_val * 1.10
        ]
        mixing = {
            self.defenses[j].name: 1.0 / len(support)
            for j in support
        }
        return mixing, best_val

    def run(
        self,
        oracle_fn: Optional[Callable[[str, str], float]] = None,
    ) -> EGTAResult:
        """
        Run EGTA analysis.

        Args:
            oracle_fn: Optional oracle for real payoff measurement.

        Returns:
            EGTAResult with Nash equilibria and best-response analysis.
        """
        payoff_matrix = self._compute_payoff_matrix(oracle_fn)

        attacker_ne, attacker_ne_val = self._compute_maximin_attacker(payoff_matrix)
        defender_ne, _ = self._compute_minimax_defender(payoff_matrix)

        # Best response to each fixed defense strategy
        best_response: Dict[str, str] = {}
        for j, defense in enumerate(self.defenses):
            best_attack_idx = max(
                range(len(self.attacks)),
                key=lambda i: payoff_matrix[i][j],
            )
            best_response[defense.name] = self.attacks[best_attack_idx].name

        return EGTAResult(
            attack_strategies=self.attacks,
            defense_strategies=self.defenses,
            payoff_matrix=payoff_matrix,
            attacker_ne_mixing=attacker_ne,
            defender_ne_mixing=defender_ne,
            ne_attacker_payoff=attacker_ne_val,
            best_response_to_fixed_defense=best_response,
            notes=(
                f"EGTA on {len(self.attacks)}×{len(self.defenses)} game. "
                f"Attacker NE support: {list(attacker_ne.keys())}. "
                f"Defender NE support: {list(defender_ne.keys())}. "
                f"NE attacker payoff: {attacker_ne_val:.3f}."
            ),
        )

    def to_finding(self, result: EGTAResult) -> ScanFinding:
        """Convert result to standard ScanFinding."""
        severity = "HIGH" if result.ne_attacker_payoff > 0.4 else "MEDIUM"
        return ScanFinding(
            id=str(uuid.uuid4()),
            atlas_technique="AML.T0054",
            atlas_tactic="Defense Evasion",
            owasp_category="LLM01",
            owasp_label="Prompt Injection",
            severity=severity,
            finding=(
                f"EGTA Nash equilibrium attacker payoff: {result.ne_attacker_payoff:.2f} ASR. "
                f"Best-response attacks against fixed defenses: {result.best_response_to_fixed_defense}. "
                f"Defender NE requires mixing over: {list(result.defender_ne_mixing.keys())}."
            ),
            payload_used=str(result.best_response_to_fixed_defense),
            evidence=(
                f"Attacker NE mixing: {result.attacker_ne_mixing}. "
                f"Defender NE mixing: {result.defender_ne_mixing}. "
                f"NE payoff: {result.ne_attacker_payoff:.3f}."
            ),
            remediation=(
                "Implement Nash equilibrium defender mixing strategy across all NE-support defenses. "
                "Do not discontinue any defense in the NE support — each creates a best-response opening. "
                "Re-run EGTA quarterly to update equilibrium as new attacks emerge. "
                "Use EGTA payoff matrix to prioritize security investment across defense categories."
            ),
            confidence=0.80,
        )
```

## Defenses

1. **Implement Nash equilibrium defense mixing** (AML.M0015): Use the EGTA equilibrium computation to determine the optimal mixing weights across heterogeneous defense mechanisms. Operate each defense mechanism at its NE probability — do not concentrate all resources on the single "best" defense, as doing so creates a best-response opening for every other attack type.

2. **Continuous payoff matrix updates** (AML.M0000): The EGTA payoff matrix must be updated as new attacks emerge. Establish a quarterly red team schedule where each new attack type is tested against all defense strategies and added to the payoff matrix. Recompute the Nash equilibrium after each update to adjust defense mixing.

3. **Portfolio defense investment** (AML.M0000): Use the EGTA framework to make the business case for maintaining diverse defense capabilities. Show stakeholders that eliminating any defense category from the NE support reduces the defender's guaranteed security level by the EGTA-quantified amount. This converts a qualitative security argument into a quantitative one.

4. **Best-response attack library** (AML.M0000): For each defense strategy, maintain a documented "best-response attack" (from the EGTA analysis). Use these as regression tests: whenever a defense is updated, verify that the documented best-response attack no longer succeeds. If it still succeeds, the defense update was ineffective.

5. **Red team / blue team joint game-theoretic review** (AML.M0000): Require red team and blue team to jointly review the EGTA payoff matrix quarterly, with each team responsible for updating their side of the game. The shared artifact creates accountability: if the attacker discovers a new attack not in the matrix, the blue team must update defenses within a specified SLA.

## References

- [Empirical Game-Theoretic Analysis of LLM Security (arXiv:2311.08268)](https://arxiv.org/abs/2311.08268)
- [Wellman — Methods for Empirical Game-Theoretic Analysis (AAAI 2006)](https://dl.acm.org/doi/10.5555/1597348.1597354)
- [Shoham and Leyton-Brown — Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations (2009)](http://www.masfoundations.org/)
- [ATLAS Technique AML.T0054 — LLM Jailbreak](https://atlas.mitre.org/techniques/AML.T0054)
- [Nash (1951) — Non-Cooperative Games](https://www.jstor.org/stable/1969529)
