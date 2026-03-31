from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a: float, b: float) -> str :
    """Useful for calclating basic arithmetics"""
    return(f"The sum of {a} and {b} is {a + b}")

@tool
def say_hello (name: str) -> str :
    """Useful for greeting a user"""
    return(f"Hello {name}, Hope you are good. Welcome to percy 101")

def main():
    model = ChatOpenAI(temperature= 0)
    
    tools = [calculator, say_hello]
    agent_executor = create_agent(model, tools)
    
    print("Welcome. I am Percy 101, your personal AI assistant.")
    print("You can easily perform basic addition or chat with me. If you want to exit, type 'quit'. ")
    
    while True:
        user_input = input("\nYou: ")
        if user_input == 'quit':
            break
        
        print("\n Assistant: ",end="")
        
        for chunk in agent_executor.stream({"messages": [HumanMessage(content= user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__" :
    main()