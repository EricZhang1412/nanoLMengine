
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_model import LinearAttentionConfig
from .model import LinearAttentionForCausalLM, LinearAttentionModel

AutoConfig.register(LinearAttentionConfig.model_type, LinearAttentionConfig, exist_ok=True)
AutoModel.register(LinearAttentionConfig, LinearAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(LinearAttentionConfig, LinearAttentionForCausalLM, exist_ok=True)

__all__ = ['LinearAttentionConfig', 'LinearAttentionForCausalLM', 'LinearAttentionModel']