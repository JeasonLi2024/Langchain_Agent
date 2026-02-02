
import os
import sys
import django
from django.conf import settings

# Add langchain-v2.0 to sys.path
langchain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if langchain_root not in sys.path:
    sys.path.append(langchain_root)

# Add project root to sys.path for Django apps
project_root = os.getenv("DJANGO_PROJECT_ROOT")
if not project_root:
    # Fallback to dev structure: Server_Project_ZH/langchain-v2.0 -> Parent is Server_Project_ZH
    project_root = os.path.dirname(langchain_root)

if project_root and project_root not in sys.path:
    sys.path.append(project_root)

# Setup Django with SQLite for testing
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test',
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': 'test_db.sqlite3',
            }
        },
        INSTALLED_APPS=[
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'user',
            'organization',
            'project',
            'projectscore', # dependency for Requirement
            'studentproject', # dependency for notification
            'authentication', # might be needed
            'notification',
            'dashboard', # signals might use it
        ],
        MIGRATION_MODULES={'audit': None, 'admin_tool': None}, # Disable problematic migrations
        USE_TZ=True,
        AUTH_USER_MODEL='user.User',
    )
    django.setup()
    
    # Create tables
    from django.core.management import call_command
    call_command('migrate', verbosity=0)

from django.contrib.auth import get_user_model
from user.models import OrganizationUser
from organization.models import Organization
from project.models import Requirement
from graph.publisher_agent import publisher_app
from langchain_core.messages import HumanMessage, AIMessage

User = get_user_model()

def test_publisher_agent():
    print("Setting up test data...")
    # Create User
    username = "agent_test_user"
    email = "agent_test@example.com"
    user, created = User.objects.get_or_create(username=username, defaults={"email": email, "password": "password"})
    if created:
        user.set_password("password")
        user.save()
    
    # Create Organization
    org_name = "Agent Test Org"
    org, created = Organization.objects.get_or_create(
        name=org_name,
        defaults={
            "organization_type": 'other',
            "other_type": 'other',
            "status": 'verified',
            "contact_person": 'Tester',
            "contact_phone": '12345678901'
        }
    )
    
    OrganizationUser.objects.get_or_create(user=user, organization=org, defaults={"permission": 'admin'})
    
    print("Starting agent interaction...")
    
    # 1. User initiates
    state = {
        "messages": [HumanMessage(content="我要发布一个Python后端项目")],
        "user_id": user.id,
        "org_id": org.id,
        "current_draft_id": 0,
        "draft_data": {},
        "next_step": "",
        "is_complete": False
    }
    
    # Run 1
    print("\n--- Round 1 ---")
    result = publisher_app.invoke(state)
    last_msg = result['messages'][-1]
    print(f"Agent: {last_msg.content}")
    
    # 2. User provides details
    print("\n--- Round 2 ---")
    user_input = "项目标题是'Python电商后台'，简介是'基于Django开发'，详细描述是'我们需要一个支持高并发的电商后台，使用Django DRF框架，需要Redis缓存和MySQL数据库。'"
    new_messages = result['messages'] + [HumanMessage(content=user_input)]
    state['messages'] = new_messages
    
    result = publisher_app.invoke(state)
    
    # Check if tool was called (save_draft)
    # The result should contain the tool call message and the tool output message if the graph executed the tool.
    # Our graph has a loop: chat -> tools -> chat.
    # So if tool was called, we should see multiple messages added.
    
    for msg in result['messages'][len(new_messages):]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"Tool Call: {msg.tool_calls}")
        elif hasattr(msg, 'content'):
            print(f"Agent/Tool Output: {msg.content}")
            
    # Check DB
    reqs = Requirement.objects.filter(title="Python电商后台", organization=org)
    if reqs.exists():
        print(f"\nSUCCESS: Requirement created! ID: {reqs.first().id}")
        req = reqs.first()
        print(f"Status: {req.status}")
        print(f"Description: {req.description}")
    else:
        print("\nFAILURE: Requirement not created.")

    # Clean up
    # user.delete()
    # org.delete()

if __name__ == "__main__":
    test_publisher_agent()
