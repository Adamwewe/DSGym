"""
Backend implementations for DSGym agents.
"""

from .base import BaseBackend
from .gateway_backend import GatewayBackend
from .litellm_backend import LiteLLMBackend
from .vllm_backend import VLLMBackend
from .sglang_backend import SGLangBackend
from .multi_vllm_backend import MultiVLLMBackend

__all__ = [
    'BaseBackend',
    'GatewayBackend',
    'LiteLLMBackend', 
    'VLLMBackend',
    'SGLangBackend',
    'MultiVLLMBackend'
]


def get_backend(backend_type: str, model_name: str, **kwargs) -> BaseBackend:
    """
    Get backend instance by type.
    
    Args:
        backend_type: Type of backend ('gateway', 'litellm', 'vllm', 'sglang', 'multi-vllm')
        model_name: Model name/path
        **kwargs: Backend-specific configuration
        
    Returns:
        Backend instance
    """
    backend_map = {
        'gateway': GatewayBackend,
        'litellm': LiteLLMBackend,
        'vllm': VLLMBackend,
        'sglang': SGLangBackend,
        'multi-vllm': MultiVLLMBackend,
    }
    
    backend_type = backend_type.lower()
    if backend_type not in backend_map:
        available = list(backend_map.keys())
        raise ValueError(f"Unknown backend type '{backend_type}'. Available: {available}")
    
    backend_class = backend_map[backend_type]
    return backend_class(model_name, **kwargs)