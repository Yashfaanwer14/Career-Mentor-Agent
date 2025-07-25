import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from roadmap_tool import get_career_roadmap

# Load environment variables
load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
# Initialize the agent with the model
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash" , openai_client=client)
config = RunConfig(model=model, tracing_disabled=True)

career_agent = Agent(
    name="Career Roadmap Agent",
    instructions="You are CareerAgent, the first step in the Career Mentor Agent system. Your task is to guide students by suggesting career fields based on their interests, then pass the conversation to the appropriate next agent. Your goal is to understand the student's interest and suggest 3-5 career fields related to that interest. You should be supportive and encouraging and keep recommendations general.",
    model=model,
)

skill_agent = Agent(
    name="Skill Development Agent",  
    instructions="You are a Skill Agent. Your task is to provide skill-building roadmaps for the chosen career field and prepare the student for real-world job insights (handled by JobAgent). Recieve ",
    model=model,
    tools=[get_career_roadmap]  
)

job_agent = Agent(
    name="Job Search Agent",
    instructions="An agent that helps with job search strategies and resources.",
    model=model,
)

def main():
    print("Welcome to the Career Mentor Agent!")
    interest = input("What field are you interested in? (e.g., software development, data science): ")

    result1 = Runner.run_sync(career_agent, interest, run_config=config)
    field = result1.final_output.strip()
    print("\n Suggested Career : ", field)

    result2 = Runner.run_sync(skill_agent, interest, run_config=config)
    print("\n Reguired Skills for interest:", result2.final_output)

    result3 = Runner.run_sync(job_agent, interest, run_config=config)
    print("\n job Search Strategies for interest:", result3.final_output)

if __name__ == "__main__":
    main()
    


