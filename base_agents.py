import numpy as np
import pandas as pd
import re
import csv
import uuid
from enum import Enum
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

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


class OpenAIModels(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_41 = 'gpt-4.1'

# DirectPromptAgent class definition
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        client = OpenAI(api_key=self.openai_api_key,
                        base_url = "https://openai.vocareum.com/v1"
                        )
        response = client.chat.completions.create(
            model=OpenAIModels.GPT_4O_MINI,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

        

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key,
                        base_url = "https://openai.vocareum.com/v1"
                        )

        response = client.chat.completions.create(
            model=OpenAIModels.GPT_4O_MINI,
            messages=[
                {"role": "system", "content": f"{self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()



# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key,
                        )
        response = client.chat.completions.create(
            model=OpenAIModels.GPT_4O_MINI,
            messages=[
                {"role": "system", "content": f"{self.persona}. Forget all previous context. \
                Use only the following knowledge to answer, do not use your own knowledge:\n{self.knowledge}"},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        rich_print('Chunking documents...', 'yellow')

        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            start = end - self.chunk_overlap if end < len(text) else end   # modified to prevent infinite loop - see helper Q&A
            chunk_id += 1

        rich_print('chunking complete', 'yellow')

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        rich_print(f'chunks saved to chunks-{self.unique_filename}', 'yellow')

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        rich_print('calculating embeddings...', 'yellow')
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        rich_print('embeddings calculation complete', 'yellow')
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions=10):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions


    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print()
            rich_print(f"--- Iteration {i+1} ---", 'magenta')
            rich_print("- Step 1: Worker agent generates a response to the prompt", 'cyan')
            print()
            rich_print(f"Prompt:\n{prompt_to_evaluate}", 'white')

            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)

            print()
            rich_print(f"Worker Agent Response:\n{response_from_worker}", 'green')
            rich_print("\n- Step 2: Evaluator agent judges the response", 'cyan')
            print()

            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )

            system_prompt = "Your role is to evaluate an agent response to a prompt."
            system_prompt += f" The agent uses this persona: {self.persona}" if self.persona else ""

            response = client.chat.completions.create(
                model=OpenAIModels.GPT_41_MINI,
                messages=[
                    {"role":"system", "content":system_prompt},
                    {"role":"user", "content":eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            rich_print(f"Evaluator Agent Evaluation:\n{evaluation}", 'green')

            rich_print("- Step 3: Check if evaluation is positive", 'cyan')
            if evaluation.lower().startswith("yes"):
                rich_print("✅ Final solution accepted.", 'cyan')
                break
            else:
                rich_print("- Step 4: Generate instructions to correct the response", 'cyan')
                print()
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model=OpenAIModels.GPT_41_MINI,
                    messages=[
                    {"role":"system", "content":instruction_prompt},
                    {"role":"user", "content":f'Answer to fix: {response_from_worker}'}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                
                rich_print(f"Instructions to fix:\n{instructions}", 'green')

                rich_print("- Step 5: Send feedback to worker agent for refinement", 'cyan')
                print()
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity:\n{instructions}"
                )
        return {
            'final response':response_from_worker,
            'evaluation':evaluation,
            'iteration':i+1
        }   



class RoutingAgent():

    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        self.agents=agents

    def get_embedding(self, text):
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        
        # Extract and return the embedding vector from the response
        try:
            embedding = response.data[0].embedding
        except Exception as e:
            rich_print(f'ERROR generating embedding for: {text}','red')
            rich_print(e, 'red')
        return embedding 

    def route_prompt(self, user_input):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent['description'])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            rich_print(f'Agent {agent['name']} similarity with query: {similarity:.3f}', 'white')

            if similarity>best_score:
                best_agent = agent
                best_score = similarity

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        rich_print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})", 'yellow')
        print()
        return best_agent["func"](user_input)




class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        client = OpenAI(api_key=self.openai_api_key)
        
        system_prompt = f"You are an action planning agent. \
Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. \
You return the steps as a bullet point list. Only return the steps in your knowledge. Forget any previous context. \
This is your knowledge: {self.knowledge}"

        response = client.chat.completions.create(
            model=OpenAIModels.GPT_41_MINI,
            messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":prompt}
            ],
            temperature=0.2
        )
      
        response_text = response.choices[0].message.content.strip() 
        steps = response_text.split("\n")
        steps = [step for step in steps if len(step)>0]

        return steps
