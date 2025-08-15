# INITIALIZATION
import os, sys

# Dynamically get the absolute path to the project root (2 levels up from phase_2/)
WORKSPACE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.exists(WORKSPACE_DIRECTORY) and WORKSPACE_DIRECTORY not in sys.path:
    sys.path.append(WORKSPACE_DIRECTORY)

from dotenv import load_dotenv
from base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import importlib.util
if importlib.util.find_spec("rich") is not None:
    from rich.console import Console
    from rich.markdown import Markdown
    console=Console()
else:
    console=None


def rich_print(text:str, color:str):
    '''use rich library for advanced printing to terminal'''
    if console:
        console.print(Markdown(text), style=f'bold {color}')
    else:
        print(text)


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open('Product-Spec-Email-Router.txt', encoding='utf-8') as f:
    product_spec = f.read()



# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. ")
knowledge_product_manager += f'\nProduct Spec:\n{product_spec}'

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# Product Manager - Evaluation Agent
persona_evaluation_agent = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria = "As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key, 
    persona=persona_evaluation_agent, 
    evaluation_criteria=evaluation_criteria, 
    worker_agent=product_manager_knowledge_agent, 
    max_interactions = 10
)


# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)


# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
program_manager_evaluation_criteria="The answer should be product features that follow the following structure: " \
                      "Feature Name: A clear, concise title that identifies the capability\n" \
                      "Description: A brief explanation of what the feature does and its purpose\n" \
                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
                      "User Benefit: How this feature creates value for the user"

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key, 
    persona=persona_program_manager_eval, 
    evaluation_criteria=program_manager_evaluation_criteria, 
    worker_agent=program_manager_knowledge_agent, 
    max_interactions = 10
)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)



# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
dev_engineer_evaluation_criteria="The answer should be tasks following this exact structure: " \
                      "Task ID: A unique identifier for tracking purposes\n" \
                      "Task Title: Brief description of the specific development work\n" \
                      "Related User Story: Reference to the parent user story\n" \
                      "Description: Detailed explanation of the technical work required\n" \
                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
                      "Estimated Effort: Time or complexity estimation\n" \
                      "Dependencies: Any tasks that must be completed first"

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key, 
    persona=persona_dev_engineer_eval, 
    evaluation_criteria=dev_engineer_evaluation_criteria, 
    worker_agent=development_engineer_knowledge_agent, 
    max_interactions = 10
)

# Routing Agent
product_manager_support_function = lambda x: product_manager_evaluation_agent.evaluate(x)
program_manager_support_function = lambda x: program_manager_evaluation_agent.evaluate(x)
development_engineer_support_function = lambda x: development_engineer_evaluation_agent.evaluate(x)

routing_agent = RoutingAgent(openai_api_key, {})
agents = [
    {
        "name": "Product Manager",
        "description": persona_product_manager,
        "func": product_manager_support_function #lambda x: product_manager_knowledge_agent.respond(x)
    },
    {
        "name": "Program Manager",
        "description": persona_program_manager,
        "func": program_manager_support_function #lambda x: program_manager_knowledge_agent.respond(x)
    },
    {
        "name": "Development Engineer",
        "description": persona_dev_engineer,
        "func": development_engineer_support_function #lambda x: development_engineer_knowledge_agent.respond(x)
    }
]

routing_agent.agents = agents

# Run the workflow

rich_print("\n*** Workflow execution started ***\n", 'yellow')
workflow_prompt = "What would the development tasks for this product be?"
rich_print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}",'yellow')
rich_print("\nDefining workflow steps from the workflow prompt", 'yellow')

steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

completed_steps = []
for i, step in enumerate(steps):
    print()
    rich_print(f'=================================== TASK {i+1} ===================================','orange_red1')
    rich_print(step, 'orange_red1')
    print()
    evaluation_output = routing_agent.route_prompt(step)
    completed_steps.append(evaluation_output)

rich_print(f'Query: {workflow_prompt}', 'magenta')
rich_print('Final response:', 'green')
rich_print(f'{evaluation_output['final response']}', 'green')