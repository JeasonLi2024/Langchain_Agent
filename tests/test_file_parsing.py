
import os
import sys
import unittest
import django

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Add langchain-v2.0 root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Sys path:", sys.path)

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Project_Zhihui.settings')
django.setup()

# Direct import since directory name has dash
from graph.file_parsing_graph import file_parsing_app

class TestFileParsing(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_doc.txt"
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试项目需求文档。\n\n项目名称：AI助手\n\n项目描述：我们需要一个能够自动回复用户消息的AI助手。技术栈包括Python和LangChain。")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_txt_parsing(self):
        initial_state = {
            "file_path": os.path.abspath(self.test_file),
            "file_name": self.test_file,
            "chunks": [],
            "summary": "",
            "success": True,
            "error": ""
        }
        
        result = file_parsing_app.invoke(initial_state)
        
        self.assertTrue(result['success'])
        self.assertTrue(len(result['chunks']) > 0)
        print(f"Summary: {result['summary']}")
        self.assertIn("AI助手", result['summary'] + result['chunks'][0])

if __name__ == '__main__':
    unittest.main()
