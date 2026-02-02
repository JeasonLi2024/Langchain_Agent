import sys
import os

# Add project root to sys.path
# Since test_graph.py is in langchain-v2.0/tests, dirname(dirname) gets us to langchain-v2.0
# We need to make sure langchain-v2.0 is in path so we can import graph.student_workflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.student_workflow import app

def test_workflow():
    print("Initializing LangGraph Workflow...")
    
    test_input = "我熟悉Python编程，并且对深度学习感兴趣，想找相关的项目。"
    student_id = 2
    
    initial_state = {
        "messages": [],
        "user_input": test_input,
        "student_id": student_id,
        "profile_data": {},
        "final_output": ""
    }
    
    print(f"\nUser Input: {test_input}\n")
    print("Running Workflow...\n")
    
    # Run the graph
    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"--- Node: {node_name} ---")
            if "messages" in state_update:
                print(state_update["messages"][-1].content)
            print("\n")

if __name__ == "__main__":
    test_workflow()
