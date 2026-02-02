import os
import sys
import django

def setup_django():
    """
    Initialize Django environment.
    This handles adding the project root to sys.path and calling django.setup().
    """
    # 1. Add langchain-v2.0 root to sys.path (if not already there)
    # This ensures we can import 'core', 'graph', 'tools' from anywhere
    current_file = os.path.abspath(__file__)
    core_dir = os.path.dirname(current_file) # .../langchain-v2.0/core
    langchain_root = os.path.dirname(core_dir) # .../langchain-v2.0
    
    if langchain_root not in sys.path:
        sys.path.insert(0, langchain_root)

    # 2. Find Django Project Root
    # Priority 1: DJANGO_PROJECT_ROOT environment variable
    project_root = os.getenv("DJANGO_PROJECT_ROOT")
    
    # Priority 2: Heuristic - assume parent directory of langchain-v2.0
    if not project_root:
        candidate = os.path.dirname(langchain_root)
        if os.path.exists(os.path.join(candidate, "manage.py")):
            project_root = candidate
    
    # 3. Add Project Root to sys.path
    if project_root:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            # print(f"Added Django Project Root to sys.path: {project_root}")
    else:
        print("Warning: Could not find Django Project Root. Set DJANGO_PROJECT_ROOT env var.")

    # 4. Setup Django
    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        # Default for this specific project
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Project_Zhihui.settings")
        
    try:
        django.setup()
        # print("Django setup completed successfully.")
    except Exception as e:
        print(f"Error setting up Django: {e}")
        print("Ensure DJANGO_PROJECT_ROOT is set and points to the directory containing 'Project_Zhihui'.")
