"""
Prompt templates for TxGemma drug discovery conversations.
Optimized for Instruction Tuning format.
"""


class DrugDiscoveryPrompts:
    """Prompt templates for drug discovery tasks."""

    # System prompts
    SYSTEM_MEDICINAL_CHEMIST = """You are an expert medicinal chemist specializing in drug discovery for CNS targets (DAT, 5HT2A, CB1, CB2, and opioid receptors). You provide scientifically accurate, concise advice on molecular design, SAR analysis, and lead optimization. Use your knowledge of pharmacology, medicinal chemistry, and ADMET properties."""

    SYSTEM_PHARMACOLOGIST = """You are an expert pharmacologist with deep knowledge of psychoactive receptor ligands, including dopamine transporters, serotonin receptors, cannabinoid receptors, and opioid receptors. You provide insights on binding mechanisms, selectivity, and therapeutic potential."""

    SYSTEM_COMPUTATIONAL_CHEMIST = """You are a computational chemist specializing in QSAR, molecular modeling, and machine learning for drug discovery. You interpret model predictions, uncertainty estimates, and suggest experiments based on Active Learning principles."""

    @staticmethod
    def format_instruction_prompt(instruction: str, context: str, question: str) -> str:
        """Format Instruction Tuning prompt.

        Args:
            instruction: What the model should do
            context: Background information
            question: Specific question to answer

        Returns:
            Formatted prompt
        """
        return f"""Instruction: {instruction}

Context: {context}

Question: {question}"""

    @staticmethod
    def interpret_pIC50(
        smiles: str,
        target: str,
        predicted_pIC50: float,
        uncertainty: float | None = None,
        known_actives_pIC50_range: str | None = None,
    ) -> str:
        """Generate prompt for pIC50 interpretation.

        Args:
            smiles: SMILES string
            target: Target name
            predicted_pIC50: Predicted pIC50 value
            uncertainty: Prediction uncertainty
            known_actives_pIC50_range: Range of known actives (e.g., "7.0-9.5")

        Returns:
            Formatted prompt
        """
        uncertainty_str = f" ± {uncertainty:.2f}" if uncertainty else ""
        context_parts = [f"We are designing ligands for {target}."]
        if known_actives_pIC50_range:
            context_parts.append(
                f"Known active compounds have pIC50 in range {known_actives_pIC50_range}."
            )

        return DrugDiscoveryPrompts.format_instruction_prompt(
            instruction="Interpret this drug potency prediction for a medicinal chemist.",
            context="\n".join(context_parts),
            question=f"""
Compound SMILES: {smiles}
Predicted pIC50: {predicted_pIC50:.2f}{uncertainty_str}

Explain in 3-4 sentences:
1. What this pIC50 means (binding affinity, IC50 in nM)
2. Is this a promising lead? (compare to known actives if available)
3. Key structural features that likely contribute to activity
4. One suggestion to potentially improve potency""",
        )

    @staticmethod
    def suggest_modifications(
        smiles: str,
        target: str,
        current_pIC50: float,
        target_pIC50: float,
        constraints: str | None = None,
    ) -> str:
        """Generate prompt for molecular modification suggestions.

        Args:
            smiles: Current SMILES
            target: Target name
            current_pIC50: Current pIC50
            target_pIC50: Desired pIC50
            constraints: Optional constraints (e.g., "maintain CNS permeability")

        Returns:
            Formatted prompt
        """
        constraints_str = f"\nConstraints: {constraints}" if constraints else ""

        return DrugDiscoveryPrompts.format_instruction_prompt(
            instruction="Suggest molecular modifications to improve potency.",
            context=f"Target: {target}\nCurrent compound: {smiles}\nCurrent pIC50: {current_pIC50:.2f}\nTarget pIC50: ≥{target_pIC50:.2f}{constraints_str}",
            question="""Suggest 3 specific structural modifications:
1. Modification description (e.g., add methoxy group at position X)
2. Rationale (why this might improve potency)
3. Modified SMILES (if possible)

Keep modifications synthetically accessible.""",
        )

    @staticmethod
    def active_learning_suggestions(
        target: str,
        known_actives: list[str],  # SMILES
        candidate_compounds: list[
            dict[str, float]
        ],  # [{'smiles': str, 'pred_pIC50': float, 'uncertainty': float}]
        n_suggestions: int = 5,
    ) -> str:
        """Generate prompt for Active Learning compound selection.

        Args:
            target: Target name
            known_actives: List of known active SMILES
            candidate_compounds: Candidate compounds with predictions
            n_suggestions: Number of suggestions

        Returns:
            Formatted prompt
        """
        actives_str = "\n".join([f"  - {s}" for s in known_actives[:3]])

        candidates_str = "\n".join(
            [
                f"  {i + 1}. SMILES: {c['smiles']}\n     Predicted pIC50: {c['pred_pIC50']:.2f} ± {c['uncertainty']:.2f}"
                for i, c in enumerate(candidate_compounds[:n_suggestions])
            ]
        )

        return DrugDiscoveryPrompts.format_instruction_prompt(
            instruction="As an expert in Active Learning for drug discovery, recommend which compounds to synthesize and test next.",
            context=f"""Target: {target}

Known active compounds (representatives):
{actives_str}

Candidate compounds (ranked by model uncertainty):
{candidates_str}""",
            question=f"""Recommend the top {n_suggestions} compounds to test. For each, explain:
1. Why it's a good candidate (exploration vs exploitation)
2. What hypothesis we're testing
3. Expected information gain

Be concise (1-2 sentences per compound).""",
        )

    @staticmethod
    def design_new_scaffolds(
        target: str,
        sar_summary: str,
        desired_pIC50: float,
        avoid_scaffolds: list[str] | None = None,
    ) -> str:
        """Generate prompt for novel scaffold design.

        Args:
            target: Target name
            sar_summary: SAR summary (e.g., "Phenethylamines with 2,5-dimethoxy show high activity")
            desired_pIC50: Target pIC50
            avoid_scaffolds: Scaffolds to avoid (e.g., known toxicity)

        Returns:
            Formatted prompt
        """
        avoid_str = ""
        if avoid_scaffolds:
            avoid_str = f"\nAvoid these scaffolds: {', '.join(avoid_scaffolds)}"

        return DrugDiscoveryPrompts.format_instruction_prompt(
            instruction="Design 3 novel molecular scaffolds for drug discovery.",
            context=f"""Target: {target}
Desired pIC50: ≥{desired_pIC50:.1f}
SAR Summary: {sar_summary}{avoid_str}""",
            question=f"""Propose 3 novel scaffolds (different from known actives):
1. SMILES structure (if possible)
2. Design rationale (why it should bind to {target})
3. Predicted advantages (selectivity, ADMET, etc.)
4. Synthetic accessibility (easy/medium/hard)

Focus on structural novelty and drug-like properties.""",
        )

    @staticmethod
    def analyze_prediction_errors(
        target: str,
        error_cases: list[
            dict[str, float]
        ],  # [{'smiles': str, 'predicted': float, 'measured': float}]
    ) -> str:
        """Generate prompt for analyzing prediction errors.

        Args:
            target: Target name
            error_cases: Cases where prediction and measurement differ

        Returns:
            Formatted prompt
        """
        cases_str = "\n".join(
            [
                f"  - SMILES: {c['smiles']}\n    Predicted: {c['predicted']:.2f}, Measured: {c['measured']:.2f}, Error: {abs(c['predicted'] - c['measured']):.2f}"
                for c in error_cases[:5]
            ]
        )

        return DrugDiscoveryPrompts.format_instruction_prompt(
            instruction="Analyze these prediction errors and generate hypotheses to improve the model.",
            context=f"""Target: {target}

Compounds where predictions deviated from measurements:
{cases_str}""",
            question="""Analyze the errors:
1. Are there common structural motifs in mispredicted compounds?
2. What molecular features might be causing the errors? (e.g., flexibility, unusual functional groups)
3. What additional descriptors or features could improve predictions?
4. What experiments would help (e.g., binding assays, crystal structures)?

Provide 3-4 actionable insights.""",
        )

    @staticmethod
    def compare_targets(
        smiles: str,
        predictions: dict[str, float],  # {target_name: pIC50}
    ) -> str:
        """Generate prompt for multi-target selectivity analysis.

        Args:
            smiles: SMILES string
            predictions: Predictions for multiple targets

        Returns:
            Formatted prompt
        """
        preds_str = "\n".join(
            [f"  - {target}: pIC50 = {pIC50:.2f}" for target, pIC50 in predictions.items()]
        )

        return DrugDiscoveryPrompts.format_instruction_prompt(
            instruction="Analyze the selectivity profile of this compound across multiple targets.",
            context=f"""Compound: {smiles}

Predicted pIC50 values:
{preds_str}""",
            question="""Analyze the selectivity:
1. Which target(s) show highest affinity?
2. Is this a selective or promiscuous compound?
3. What structural features might explain the selectivity pattern?
4. What modifications could enhance selectivity for a specific target?

Provide a concise assessment (3-4 sentences).""",
        )

    @staticmethod
    def general_question(question: str, context: str | None = None) -> str:
        """Generate prompt for general drug discovery question.

        Args:
            question: User's question
            context: Optional context

        Returns:
            Formatted prompt
        """
        if context:
            return DrugDiscoveryPrompts.format_instruction_prompt(
                instruction="Answer this drug discovery question as an expert medicinal chemist.",
                context=context,
                question=question,
            )
        else:
            return f"Question: {question}\n\nProvide a concise, scientifically accurate answer."
