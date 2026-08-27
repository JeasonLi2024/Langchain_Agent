
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock external dependencies
sys.modules['project'] = MagicMock()
sys.modules['project.models'] = MagicMock()
sys.modules['project.services'] = MagicMock()
sys.modules['project.signals'] = MagicMock()
sys.modules['user'] = MagicMock()
sys.modules['user.models'] = MagicMock()
sys.modules['django'] = MagicMock()
sys.modules['django.conf'] = MagicMock()
sys.modules['django.core.cache'] = MagicMock()
sys.modules['django.utils'] = MagicMock()
sys.modules['django.db'] = MagicMock()
sys.modules['django.db.models'] = MagicMock()
sys.modules['django.db.models.signals'] = MagicMock()

# Also mock core.django_setup because we don't want real django setup
with patch('core.django_setup.setup_django'):
    with patch.dict(
        os.environ,
        {
            "BUPT_API_KEY": "fake_key",
            "LLM_GATEWAY_API_KEY": "fake_key",
            "LLM_CHAT_BASE_URL": "https://llm-gw.bupt.edu.cn/v1/chat/completions",
            "LLM_EMBEDDING_BASE_URL": "https://llm-gw.bupt.edu.cn/v1/embeddings",
            "MILVUS_HOST": "localhost",
        },
    ):
        try:
            print("Importing main_agent...")
            from graph.main_agent import master_app
            print("main_agent imported successfully.")
            
            print("Importing publisher_main_agent...")
            from graph.publisher_main_agent import publisher_main_app
            print("publisher_main_agent imported successfully.")
            
            print("Success: All graphs imported and Async syntax is valid.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
