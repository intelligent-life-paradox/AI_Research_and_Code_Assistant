
import sys
import warnings
from dotenv import load_dotenv
import os 
load_dotenv()

from crew_diffusion.crew import CreateCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew with a user question about diffusion models.

    """
    user_input = os.environ.get("USER_INPUT") or input("Ask anything about diff. models or codes: ")
    
    
    user_input = input("Ask anything about diff. models or codes: ")
    
    inputs = {
        "user_input": user_input
    }

    try:
        result = CreateCrew().crew().kickoff(inputs=inputs)
        print("\n=== RESULT ===")
        print(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "user_input": "Explain the forward diffusion process in DDPM"
    }
    try:
        CreateCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CreateCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "user_input": "Explain the forward diffusion process in DDPM"
    }

    try:
        CreateCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
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
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")


if __name__ == "__main__":
    run()