import sys
import os
import asyncio
from langchain_core.messages import HumanMessage

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.main_agent import master_app

async def test_full_conversation():
    print("Initializing Master Agent (A)...")
    
    # Simulate a conversation session
    conversation_history = []
    
    # Turn 1: Greeting (Chat Intent)
    print("\n--- Turn 1: User says '你好' ---")
    inputs = {"messages": [HumanMessage(content="你好")]}
    
    # We need to capture the output of the 'chat_node'. 
    # In astream_events, we look for 'on_chat_model_stream'.
    # However, sometimes if the model is very fast or cached, it might come as 'on_chat_model_end' output.
    # But usually 'on_chat_model_stream' works if streaming is enabled on the model.
    # Note: Config.get_utility_llm() might not have streaming=True by default in config.py
    
    async for event in master_app.astream_events(inputs, version="v1"):
        kind = event["event"]
        
        # Debug print to see what events are happening
        # print(f"[DEBUG Event] {kind} - {event['name']}") 
        
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
                
    print("\n")
    
    # Turn 2: Recommendation (Recommend Intent)
    print("\n--- Turn 2: User asks for recommendation ---")
    # In a real app, we would append to history. For this test, we construct input with history if needed,
    # but our router mainly looks at the last message.
    inputs = {"messages": [HumanMessage(content="我熟悉Python编程，并且对深度学习感兴趣，想找相关的项目。")]}
    
    print("(Agent is thinking...)\n")
    async for event in master_app.astream_events(inputs, version="v1"):
        kind = event["event"]
        name = event["name"]
        
        # We want to see the "Thinking" process from the subgraph
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
                
        # Debug: Show routing
        elif kind == "on_chain_start" and name in ["router", "chat_node", "recommendation_flow"]:
            print(f"\n[System] Routing to: {name}\n")

if __name__ == "__main__":
    asyncio.run(test_full_conversation())
