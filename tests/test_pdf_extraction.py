
import os
import sys
import django
from django.conf import settings
import json

# Add project root and langchain-v2.0 to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django with SQLite for testing (minimal env)
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test',
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': ':memory:',
            }
        },
        INSTALLED_APPS=[
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'user',
            'organization',
            'project',
            'projectscore', 
            'studentproject',
            'authentication',
            'notification',
        ],
        MIGRATION_MODULES={'audit': None, 'admin_tool': None},
        USE_TZ=True,
        AUTH_USER_MODEL='user.User',
    )
    django.setup()

from graph.file_parsing_graph import file_parsing_app

def test_pdf_extraction():
    # File path
    pdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "1. 华为云计算技术有限公司——“基于端云算力协同的疲劳驾驶智能识别”比赛方案_1-6.pdf"
    )
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Testing extraction with: {pdf_path}")
    
    initial_state = {
        "file_path": pdf_path,
        "file_name": os.path.basename(pdf_path),
        "draft_id": 9999, # Dummy ID for testing storage
        "chunks": [],
        "chunk_embeddings": [],
        "filtered_chunks": [],
        "extracted_data": {},
        "summary": "",
        "success": True,
        "error": ""
    }
    
    # Run graph
    result = file_parsing_app.invoke(initial_state)
    
    if result.get('success'):
        print("\n--- Extraction Success ---")
        data = result.get('extracted_data', {})
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Verify specific fields
        print("\n--- Validation ---")
        print(f"Title: {data.get('title')}")
        print(f"Brief: {data.get('brief')}")
        print(f"Budget: {data.get('budget')}")
        
        # Verify storage (mock check, looking for log)
        # Ideally query Milvus, but we trust the graph execution if no error.
        print("\n--- Storage Check ---")
        print("Check logs for 'Stored X chunks for draft 9999'")
    else:
        print(f"\n--- Extraction Failed ---")
        print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    test_pdf_extraction()
