#!/usr/bin/env python
import sys
import warnings
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from crew_diffusion.crew import CreateCrew
except ImportError:
    from src.crew_diffusion.crew import CreateCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run(inputs=None):
    user_query = ""

    if isinstance(inputs, str):
        user_query = inputs
    elif isinstance(inputs, dict):
        user_query = inputs.get("user_input", "")
    elif inputs is None:
        user_query = os.environ.get("USER_INPUT")
        
    if not user_query:
        try:
            user_query = input("Ask anything about diff. models or codes: ")
        except EOFError:
            print("[ERROR] No input provided in non-interactive environment.")
            return "Error: No input provided."

    crew_inputs = {
        "user_input": user_query
    }

    try:
        crew_instance = CreateCrew().crew()
        result = crew_instance.kickoff(inputs=crew_inputs)
        
        print("\n < RESULT > \n")
        print(result)
        
        return str(result)

    except Exception as e:
        error_message = f"An error occurred while running the crew: {e}"
        print(error_message)
        return error_message

def train():
    inputs = {
        "user_input": "Explain the forward diffusion process in DDPM"
    }
    try:
        n_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        filename = sys.argv[2] if len(sys.argv) > 2 else "trained_agents.pkl"

        CreateCrew().crew().train(
            n_iterations=n_iterations,
            filename=filename,
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    try:
        if len(sys.argv) < 2:
            raise ValueError("Task ID is required for replay.")
            
        CreateCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    inputs = {
        "user_input": "Explain the forward diffusion process in DDPM"
    }

    try:
        n_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        eval_llm = sys.argv[2] if len(sys.argv) > 2 else "gpt-3.5-turbo"

        CreateCrew().crew().test(
            n_iterations=n_iterations,
            eval_llm=eval_llm,
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "user_input": trigger_payload.get("user_input", "")
    }

    try:
        result = CreateCrew().crew().kickoff(inputs=inputs)
        return str(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "train":
            sys.argv.pop(0) 
            train()
        elif command == "replay":
            sys.argv.pop(0)
            replay()
        elif command == "test":
            sys.argv.pop(0)
            test()
        elif command == "run_trigger":
            sys.argv.pop(0)
            run_with_trigger()
        else:
            run()
    else:
        run()
