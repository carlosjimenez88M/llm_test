'''
RAG system executor program, LLM for a medical services startup case
This program is based on being able to track and systematize the solution
By : Carlos Daniel Jiménez
Date : 2024-01-14
'''

#====================#
# ---- libraries --- #
#====================#
import os
import sys
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import mlflow
import wandb
import pandas as pd
from modules.interaction_tracker import InteractionTracker
from modules.vector_store import create_vector_store
from modules.react_agent import create_react_agent
from modules.evaluation import calculate_metrics, generate_evaluation_report


def setup_logging_directories(cfg: DictConfig) -> Dict[str, str]:
    """
    Set up and validate all necessary directories for logging.
    """
    paths = {}
    required_paths = ['reference_data', 'current_data', 'logs', 'drift_report']
    for path_key in required_paths:
        path = to_absolute_path(cfg.paths[path_key])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        paths[path_key] = path
    return paths


def init_tracking_systems(cfg: DictConfig) -> None:
    """
    Initialize and configure MLflow and Weights & Biases tracking.
    """
    # Initialize W&B
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=dict(cfg)
    )


def validate_environment() -> str:
    """
    Validate environment variables and return OpenAI API key.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")
    return openai_api_key


def process_queries(agent, queries: List[str], tracker: InteractionTracker,
                   truth_dict: Dict[str, str]) -> tuple[List[str], List[str], List[str]]:
    """
    Process a list of queries through the agent and track interactions.

    Parameters
    ----------
    agent : Agent
        The initialized agent to process queries
    queries : List[str]
        List of queries to process
    tracker : InteractionTracker
        Tracker for logging interactions
    truth_dict : Dict[str, str]
        Dictionary mapping queries to ground truth responses

    Returns
    -------
    tuple[List[str], List[str], List[str]]
        Tuple containing (processed_queries, responses, ground_truth)
    """
    responses = []
    processed_queries = []
    ground_truth = []

    for query in queries:
        try:
            response = agent.run(query)
            tracker.log_interaction(query, response)

            processed_queries.append(query)
            responses.append(response)
            ground_truth.append(truth_dict.get(query, ""))

            # Log to W&B
            wandb.log({
                "user_query": query,
                "agent_response": response,
                "ground_truth": truth_dict.get(query, ""),
                "response_length": len(response)
            })

        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")
            continue

    return processed_queries, responses, ground_truth


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running the LLM RAG system.
    """
    try:
        # Validate environment and setup
        openai_api_key = validate_environment()
        paths = setup_logging_directories(cfg)

        # Initialize W&B first
        init_tracking_systems(cfg)

        # Handle MLflow run management
        active_run = mlflow.active_run()
        if active_run:
            run = active_run
        else:
            run = mlflow.start_run()

        try:
            # Log parameters
            mlflow.log_params(dict(cfg))

            # Create vector store
            print("Creating vector store...")
            vector_store = create_vector_store(
                data_path=paths['reference_data'],
                openai_api_key=openai_api_key
            )

            # Create React Agent
            print("Initializing React Agent...")
            agent = create_react_agent(
                vector_store=vector_store,
                openai_api_key=openai_api_key
            )

            # Configure interaction tracking
            tracker = InteractionTracker(log_file=paths['logs'])

            # Load ground truth data
            reference_data = pd.read_csv(paths['reference_data'])
            truth_dict = dict(zip(reference_data['query'], reference_data['response']))

            # Define test queries
            user_queries = [
                "What does my insurance cover?",
                "How can I file a claim?",
                "What is the copay for visits?"
            ]

            # Process queries and collect responses
            print("Processing queries...")
            queries, responses, ground_truth = process_queries(
                agent, user_queries, tracker, truth_dict
            )

            # Generate evaluation metrics and report
            print("Generating evaluation report...")
            metrics, report = generate_evaluation_report(
                queries=queries,
                predictions=responses,
                ground_truth=ground_truth
            )

            # Save evaluation report
            report_path = paths['drift_report'].replace('.html', '.txt')
            with open(report_path, 'w') as f:
                f.write(report)

            # Log metrics and artifacts
            print("Logging metrics and artifacts...")
            mlflow.log_metrics(metrics)
            wandb.log(metrics)

            # Save logs and artifacts
            tracker.save_log()
            mlflow.log_artifact(paths['logs'])
            mlflow.log_artifact(report_path)

            # Calculate and log conversation success metrics
            conversation_metrics = {
                "total_queries": len(queries),
                "successful_responses": len(responses)
            }

            success_rate = (conversation_metrics["successful_responses"] /
                          conversation_metrics["total_queries"])

            wandb.log({
                "conversation_success_rate": success_rate,
                **conversation_metrics
            })

            print("Evaluation complete!")

        finally:
            # Only end the run if we started it
            if not active_run:
                mlflow.end_run()

    except Exception as e:
        print(f"Error in main execution: {str(e)}", file=sys.stderr)
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()