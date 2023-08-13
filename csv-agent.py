from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = ""


if __name__ == '__main__':  
    
    agent = create_csv_agent(
    ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"),
    ["4.csv", "3.csv"],
    verbose=True)

    print(agent.run("summarize both of these files"))