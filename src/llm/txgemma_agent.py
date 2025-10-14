"""
TxGemma-9B agent for drug discovery conversations via Ollama.
Optimized for RTX3060 with 4bit quantization.
"""

import logging
from typing import Dict, List, Optional, Tuple
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not installed. Install with: pip install ollama")


class TxGemmaAgent:
    """TxGemma-9B conversation agent for drug discovery.
    
    Connects to Ollama to use TxGemma-9B (Chat or Predict variant).
    Provides natural language interface for:
    - Molecular design suggestions
    - pIC50 prediction interpretation
    - Active Learning compound suggestions
    - Experimental result analysis
    """
    
    def __init__(
        self,
        model_name: str = 'txgemma:9b',
        temperature: float = 0.7,
        max_tokens: int = 1024,
        context_window: int = 8192
    ):
        """Initialize TxGemma agent.
        
        Args:
            model_name: Ollama model name ('txgemma:9b' or 'txgemma:9b-chat')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens per response
            context_window: Context window size
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not installed. Run: pip install ollama")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.logger = logging.getLogger(__name__)
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 turns
        
        # Check if model is available
        self._check_model()
        
        self.logger.info(f"TxGemma agent initialized: model={model_name}, temp={temperature}")
    
    def _check_model(self) -> None:
        """Check if TxGemma model is available in Ollama."""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model_name not in model_names:
                self.logger.warning(
                    f"Model {self.model_name} not found in Ollama. "
                    f"Please run: ollama pull {self.model_name}"
                )
        except Exception as e:
            self.logger.warning(f"Could not check Ollama models: {e}")
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """Send a message to TxGemma and get response.
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt to set context
            stream: Whether to stream response
            
        Returns:
            TxGemma's response
        """
        # Build messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add conversation history
        for turn in self.history[-self.max_history:]:
            messages.append(turn)
        
        # Add current user message
        messages.append({
            'role': 'user',
            'content': user_message
        })
        
        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                }
            )
            
            assistant_message = response['message']['content']
            
            # Update history
            self.history.append({'role': 'user', 'content': user_message})
            self.history.append({'role': 'assistant', 'content': assistant_message})
            
            # Trim history if too long
            if len(self.history) > self.max_history * 2:
                self.history = self.history[-self.max_history * 2:]
            
            return assistant_message
            
        except Exception as e:
            self.logger.error(f"Ollama chat error: {e}")
            return f"Error: {str(e)}"
    
    def predict_compound_pIC50(
        self,
        smiles: str,
        target: str,
        prediction: float,
        uncertainty: Optional[float] = None
    ) -> str:
        """Get natural language explanation of pIC50 prediction.
        
        Args:
            smiles: SMILES string
            target: Target name (e.g., 'DAT', '5HT2A')
            prediction: Predicted pIC50 value
            uncertainty: Prediction uncertainty (optional)
            
        Returns:
            Natural language explanation
        """
        uncertainty_str = f" with uncertainty ±{uncertainty:.2f}" if uncertainty else ""
        
        prompt = f"""
Instruction: Interpret the following drug discovery prediction result for a chemist.

Context: We are designing ligands for {target} receptor.

Prediction Result:
- SMILES: {smiles}
- Predicted pIC50: {prediction:.2f}{uncertainty_str}
- Target: {target}

Question: Please explain:
1. What does this pIC50 value mean in terms of binding affinity?
2. Is this a promising lead compound?
3. What molecular features might contribute to this activity?
4. What modifications could potentially improve the activity?

Provide a concise, scientifically accurate explanation in 3-4 sentences.
"""
        return self.chat(prompt)
    
    def suggest_next_compounds(
        self,
        target: str,
        known_actives: List[Tuple[str, float]],
        uncertain_compounds: List[Tuple[str, float, float]],
        n_suggestions: int = 5
    ) -> str:
        """Get Active Learning suggestions in natural language.
        
        Args:
            target: Target name
            known_actives: List of (SMILES, pIC50) for known actives
            uncertain_compounds: List of (SMILES, pred_pIC50, uncertainty) for candidates
            n_suggestions: Number of suggestions
            
        Returns:
            Natural language suggestions
        """
        actives_str = "\n".join([f"  - {s} (pIC50={p:.2f})" for s, p in known_actives[:5]])
        candidates_str = "\n".join([
            f"  {i+1}. {s} (predicted pIC50={p:.2f}, uncertainty=±{u:.2f})"
            for i, (s, p, u) in enumerate(uncertain_compounds[:n_suggestions])
        ])
        
        prompt = f"""
Instruction: As a medicinal chemist, suggest which compounds to synthesize next for Active Learning.

Context: We are optimizing {target} ligands. Known active compounds:
{actives_str}

Candidate compounds (ranked by uncertainty):
{candidates_str}

Question: Which {n_suggestions} compounds should we synthesize and test next? For each:
1. Why is it a good candidate?
2. What hypothesis are we testing?
3. What molecular features make it interesting?

Provide concise recommendations for each compound.
"""
        return self.chat(prompt)
    
    def design_new_molecules(
        self,
        target: str,
        sar_info: str,
        desired_properties: Dict[str, float],
        n_suggestions: int = 3
    ) -> str:
        """Get molecular design suggestions.
        
        Args:
            target: Target name
            sar_info: Structure-Activity Relationship information
            desired_properties: Desired property ranges (e.g., {'pIC50': 8.0, 'LogP': 3.5})
            n_suggestions: Number of molecular suggestions
            
        Returns:
            Natural language molecular design suggestions
        """
        props_str = ", ".join([f"{k}={v}" for k, v in desired_properties.items()])
        
        prompt = f"""
Instruction: Design {n_suggestions} new molecular structures for {target} receptor.

Context:
- Target: {target}
- SAR Information: {sar_info}
- Desired Properties: {props_str}

Question: Suggest {n_suggestions} novel molecular structures (provide SMILES if possible) that:
1. Likely have high activity against {target}
2. Meet the desired property criteria
3. Are synthetically accessible
4. Have structural novelty

For each suggestion, explain the design rationale.
"""
        return self.chat(prompt)
    
    def analyze_experimental_results(
        self,
        predictions: List[Tuple[str, float]],
        measurements: List[Tuple[str, float]],
        target: str
    ) -> str:
        """Analyze discrepancies between predictions and measurements.
        
        Args:
            predictions: List of (SMILES, predicted_pIC50)
            measurements: List of (SMILES, measured_pIC50)
            target: Target name
            
        Returns:
            Analysis of prediction accuracy and hypotheses
        """
        results = []
        for (s_pred, pred), (s_meas, meas) in zip(predictions, measurements):
            if s_pred == s_meas:
                error = abs(pred - meas)
                results.append(f"  - {s_pred}: predicted={pred:.2f}, measured={meas:.2f}, error={error:.2f}")
        
        results_str = "\n".join(results[:5])
        
        prompt = f"""
Instruction: Analyze the prediction accuracy for {target} ligands and generate hypotheses.

Context: Comparison of predictions vs. experimental measurements:
{results_str}

Question:
1. What patterns do you see in the prediction errors?
2. Which molecular features might be causing mispredictions?
3. How can we improve the model?
4. What additional experiments would be informative?

Provide a scientific analysis in 4-5 sentences.
"""
        return self.chat(prompt)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        self.logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.history.copy()
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation history to file.
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation history from file.
        
        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
        self.logger.info(f"Conversation loaded from {filepath}")

