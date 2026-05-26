"""
Interactive CLI for TxGemma drug discovery conversations.
Integrates pIC50 prediction, Active Learning, and natural language dialogue.
"""

import argparse
import logging
from typing import Optional

from .prompts import DrugDiscoveryPrompts
from .txgemma_agent import OLLAMA_AVAILABLE, TxGemmaAgent

# Import predictor if available
try:
    from ..active_learning.selector import ActiveLearningSelector
    from ..models.uncertainty import UncertaintyEstimator

    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False


class InteractiveDrugDiscoveryCLI:
    """Interactive CLI for drug discovery with TxGemma."""

    def __init__(
        self,
        model_name: str = "txgemma:9b",
        predictor=None,
        uncertainty_estimator: Optional["UncertaintyEstimator"] = None,
        al_selector: Optional["ActiveLearningSelector"] = None,
    ):
        """Initialize interactive CLI.

        Args:
            model_name: TxGemma model name
            predictor: Trained pIC50 predictor (optional)
            uncertainty_estimator: Uncertainty estimator (optional)
            al_selector: Active Learning selector (optional)
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not installed. Run: pip install ollama")

        self.agent = TxGemmaAgent(model_name=model_name)
        self.prompts = DrugDiscoveryPrompts()
        self.predictor = predictor
        self.uncertainty_estimator = uncertainty_estimator
        self.al_selector = al_selector

        self.logger = logging.getLogger(__name__)
        self.logger.info("Interactive Drug Discovery CLI initialized")

    def print_welcome(self) -> None:
        """Print welcome message."""
        print("\n" + "=" * 70)
        print("🧪 TxGemma Interactive Drug Discovery CLI")
        print("=" * 70)
        print("\nWelcome! I'm your AI assistant for drug discovery.")
        print("I can help with:")
        print("  - Molecular design and optimization")
        print("  - pIC50 prediction interpretation")
        print("  - Active Learning suggestions")
        print("  - SAR analysis and hypothesis generation")
        print("\nCommands:")
        print("  /predict <SMILES>   - Predict pIC50 and explain")
        print("  /suggest            - Suggest next compounds (Active Learning)")
        print("  /design             - Design new molecules")
        print("  /help               - Show this help")
        print("  /clear              - Clear conversation history")
        print("  /save <file>        - Save conversation")
        print("  /quit               - Exit")
        print("\nType your question or use a command to start!")
        print("=" * 70 + "\n")

    def handle_predict(self, smiles: str) -> None:
        """Handle /predict command.

        Args:
            smiles: SMILES string to predict
        """
        if not self.predictor:
            print("❌ Predictor not loaded. Please load a trained model first.")
            return

        print(f"\n🔬 Predicting pIC50 for: {smiles}")
        print("⏳ Running prediction...")

        try:
            # Make prediction
            prediction, confidence = self.predictor.predict(smiles)

            if prediction is None:
                print("❌ Prediction failed. Invalid SMILES?")
                return

            uncertainty = confidence.get("std", 0.0) if confidence else None

            print("\n📊 Prediction Results:")
            print(f"  Predicted pIC50: {prediction:.2f}")
            if uncertainty:
                print(f"  Uncertainty: ±{uncertainty:.2f}")
                print(f"  95% CI: [{confidence['lower']:.2f}, {confidence['upper']:.2f}]")

            # Get TxGemma interpretation
            print("\n🤖 TxGemma Analysis:")
            print("-" * 70)

            response = self.agent.predict_compound_pIC50(
                smiles=smiles,
                target="DAT",  # TODO: Make target configurable
                prediction=prediction,
                uncertainty=uncertainty,
            )
            print(response)
            print("-" * 70)

        except Exception as e:
            print(f"❌ Error: {e}")
            self.logger.error(f"Prediction error: {e}", exc_info=True)

    def handle_suggest(self) -> None:
        """Handle /suggest command for Active Learning."""
        if not self.al_selector:
            print("❌ Active Learning not configured.")
            return

        print("\n🎯 Generating Active Learning suggestions...")
        print("⏳ This may take a moment...")

        try:
            # TODO: Implement actual Active Learning pipeline
            # For now, show placeholder
            print("\n📋 Top 5 Compounds to Synthesize Next:")
            print("-" * 70)

            response = self.agent.chat(
                """Suggest 5 molecular structures (SMILES) to synthesize next for DAT inhibitor discovery.
                
We have tested:
- Methylphenidate (pIC50=7.2)
- Cocaine analogs (pIC50=7.5-8.0)
- Tropanes (pIC50=6.8-7.5)

Suggest diverse compounds with high predicted activity and explain the rationale.""",
                system_prompt=self.prompts.SYSTEM_MEDICINAL_CHEMIST,
            )
            print(response)
            print("-" * 70)

        except Exception as e:
            print(f"❌ Error: {e}")
            self.logger.error(f"Suggestion error: {e}", exc_info=True)

    def handle_design(self) -> None:
        """Handle /design command for molecular design."""
        print("\n🎨 Molecular Design Assistant")
        print("-" * 70)

        target = input("Target receptor (e.g., DAT, 5HT2A): ").strip() or "DAT"
        sar_info = (
            input("SAR information (press Enter for general): ").strip() or "General DAT inhibitors"
        )
        desired_pIC50 = input("Desired pIC50 (default 8.0): ").strip()
        desired_pIC50 = float(desired_pIC50) if desired_pIC50 else 8.0

        print(f"\n⏳ Designing molecules for {target}...")

        try:
            response = self.agent.design_new_molecules(
                target=target,
                sar_info=sar_info,
                desired_properties={"pIC50": desired_pIC50},
                n_suggestions=3,
            )

            print("\n🧬 Design Suggestions:")
            print("-" * 70)
            print(response)
            print("-" * 70)

        except Exception as e:
            print(f"❌ Error: {e}")
            self.logger.error(f"Design error: {e}", exc_info=True)

    def handle_help(self) -> None:
        """Handle /help command."""
        self.print_welcome()

    def handle_clear(self) -> None:
        """Handle /clear command."""
        self.agent.clear_history()
        print("✅ Conversation history cleared.")

    def handle_save(self, filepath: str) -> None:
        """Handle /save command.

        Args:
            filepath: Path to save conversation
        """
        try:
            self.agent.save_conversation(filepath)
            print(f"✅ Conversation saved to {filepath}")
        except Exception as e:
            print(f"❌ Save failed: {e}")

    def process_command(self, user_input: str) -> bool:
        """Process command input.

        Args:
            user_input: User input string

        Returns:
            True if should continue, False if should quit
        """
        user_input = user_input.strip()

        if not user_input:
            return True

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "/quit" or command == "/exit" or command == "/q":
                print("\n👋 Goodbye! Happy discovering!")
                return False

            elif command == "/predict":
                if not args:
                    print("Usage: /predict <SMILES>")
                else:
                    self.handle_predict(args)

            elif command == "/suggest":
                self.handle_suggest()

            elif command == "/design":
                self.handle_design()

            elif command == "/help" or command == "/?":
                self.handle_help()

            elif command == "/clear":
                self.handle_clear()

            elif command == "/save":
                if not args:
                    print("Usage: /save <filepath>")
                else:
                    self.handle_save(args)

            else:
                print(f"❌ Unknown command: {command}")
                print("Type /help for available commands.")

        else:
            # General conversation
            try:
                print("\n🤖 TxGemma:")
                print("-" * 70)
                response = self.agent.chat(
                    user_input, system_prompt=self.prompts.SYSTEM_MEDICINAL_CHEMIST
                )
                print(response)
                print("-" * 70 + "\n")
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"Chat error: {e}", exc_info=True)

        return True

    def run(self) -> None:
        """Run interactive CLI loop."""
        self.print_welcome()

        try:
            while True:
                user_input = input("\n💬 You: ").strip()

                should_continue = self.process_command(user_input)
                if not should_continue:
                    break

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
        except EOFError:
            print("\n\n👋 EOF. Goodbye!")


def main():
    """Main entry point for interactive CLI."""
    parser = argparse.ArgumentParser(description="TxGemma Interactive Drug Discovery CLI")
    parser.add_argument("--model", default="txgemma:9b", help="TxGemma model name")
    parser.add_argument("--predictor-model", help="Path to trained pIC50 predictor model")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load predictor if provided
    predictor = None
    if args.predictor_model and PREDICTOR_AVAILABLE:
        try:
            # TODO: Load predictor model
            logging.info(f"Loading predictor from {args.predictor_model}")
        except Exception as e:
            logging.error(f"Failed to load predictor: {e}")

    # Initialize CLI
    cli = InteractiveDrugDiscoveryCLI(model_name=args.model, predictor=predictor)

    # Run interactive loop
    cli.run()


if __name__ == "__main__":
    main()
