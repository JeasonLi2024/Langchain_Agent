
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock external dependencies
sys.modules['project'] = MagicMock()
sys.modules['project.models'] = MagicMock()
sys.modules['project.services'] = MagicMock()
sys.modules['user'] = MagicMock()
sys.modules['user.models'] = MagicMock()
sys.modules['django'] = MagicMock()
sys.modules['django.conf'] = MagicMock()
sys.modules['django.core.cache'] = MagicMock()

# Also mock core.django_setup because we don't want real django setup
with patch('core.django_setup.setup_django'):
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake_key", "MILVUS_HOST": "localhost"}):
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
