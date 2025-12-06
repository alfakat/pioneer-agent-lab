# pioneer-agent-lab
My first RAG-powered agent. A learning project where I explored core agent concepts, document retrieval, semantic search, and pipeline building.

# How it started
I have been following DataTalksClub for some time and enrolled in their '7-Day AI Agents Email Crash Course' to get familiar with the steps required to build an effective agent.

# Few words about agents 
I found an article on ByteByteGo that provides the clearest explanation of what an AI agent is and the different types of agents:

AI agent at work, a system that doesn’t just respond to commands but actively works toward accomplishing our goals.

Traditional software programs follow rigid, predefined instructions. A calculator app performs the exact operation we specify. A search engine returns results for our query, but doesn’t take further action. These programs are powerful, but they’re fundamentally passive. They wait for our explicit commands and execute them precisely as programmed.

AI agents represent something fundamentally different. They can perceive their environment, make decisions based on what they observe, use various tools to accomplish tasks, and adapt their approach when things don’t go as planned. They exhibit a degree of autonomy that traditional software simply doesn’t have.

<img width="1456" height="1852" alt="image" src="https://github.com/user-attachments/assets/9bd6c409-bdb3-4834-8da3-50405851243d" />

Four key characteristics define what makes something an AI agent:

- First is autonomy, or the ability to operate without constant human intervention and make decisions independently.
- Second is reactivity or responding appropriately to changes in the environment and new information.
- Third is proactiveness, which is about taking initiative to achieve goals rather than just reacting to immediate stimuli.
- Fourth is social ability, or the capacity to interact with other agents or humans through some form of communication.

The diagram below shows the basic setup of an AI agent.

<img width="1456" height="889" alt="image" src="https://github.com/user-attachments/assets/493ee8e4-a467-4f23-8f4f-52811ee65776" />

Types of AI Agents

<img width="1456" height="998" alt="image" src="https://github.com/user-attachments/assets/ea4af6f0-3bbd-4ac5-918c-da002bf7875a" />

Simple Reflex Agents
Simple reflex agents are the most basic type, operating on straightforward condition-action rules. These agents perceive the current state of their environment and respond with predetermined actions based on pattern matching. Think of a thermostat. When the temperature drops below the set threshold, it turns on the heating. However, when the temperature rises above another threshold, it turns off the heating. In the software world, a basic customer service chatbot that recognizes keywords like “refund” and responds with a preset message about the refund policy is a simple reflex agent. These agents are fast, predictable, and easy to build.

<img width="1456" height="889" alt="image" src="https://github.com/user-attachments/assets/a619d6aa-1845-4a61-a34a-7d6ba7cf763e" />

Model-Based Agents
Model-based agents represent a significant step up in sophistication because they maintain an internal representation of the world that they cannot directly perceive. Consider a robot vacuum cleaner navigating a home. It cannot see the entire floor layout at once, but as it moves around, it builds a mental map of where furniture is located, which areas it has already cleaned, and where obstacles exist. This internal model allows it to plan efficient cleaning routes and avoid repeatedly bumping into the same obstacles.

<img width="1456" height="889" alt="image" src="https://github.com/user-attachments/assets/56139262-f220-4206-aa8b-2de424292760" />

Goal-Based Agents
Goal-based agents take things further by explicitly working toward specific objectives rather than simply reacting to current conditions. Imagine an AI agent helping someone prepare for a job interview at a technology company. The agent understands that the overarching goal is thorough interview preparation. It reasons about what this requires: researching the specific company and role, identifying common interview questions for that position, helping craft strong answers based on the candidate’s background, suggesting technical topics to review, and creating a study schedule for the remaining time. Each action the agent takes is evaluated based on whether it contributes to the goal of being well-prepared.

<img width="1456" height="1126" alt="image" src="https://github.com/user-attachments/assets/c229b862-d0dd-433f-b6cd-57b0098c65ce" />

Utility-Based Agents
While goal-based agents treat objectives as binary (achieved or not achieved), utility-based agents work with a more nuanced measure of success. These agents use a utility function that quantifies how desirable different outcomes are, allowing them to make decisions that optimize overall satisfaction rather than just checking boxes. Consider an agent planning our vacation.
A goal-based agent might successfully find a trip within our budget and dates. A utility-based agent, however, considers multiple factors simultaneously: cost, travel time, accommodation quality, proximity to attractions, weather conditions, and available activities. When choosing between a budget hotel requiring an hour’s commute to attractions versus a pricier but centrally located option, the utility-based agent weighs these trade-offs against our preferences and constraints. It doesn’t just find a solution, but tries to find the best solution according to a holistic evaluation of what matters most.

<img width="1456" height="1126" alt="image" src="https://github.com/user-attachments/assets/dee1afab-b136-4c0d-8e8b-3babef47b60d" />

Learning Agents
Learning agents represent the most advanced category because they improve their performance over time through experience.
A learning agent has several components working together:
- A performance element that selects actions.
- A critic that provides feedback on how well the agent is doing.
- A learning element that makes improvements based on this feedback.
- A problem generator that encourages exploration of new approaches.

In practice, this might look like a customer service agent who tracks which responses lead to customer satisfaction and which lead to escalation or complaints.

Over time, it learns which communication styles work best for different types of issues, which solutions are most effective for common problems, and when to escalate matters to human representatives. The agent might maintain a long-term memory of successful strategies, user preferences, and effective problem-solving patterns.

This ability to learn and adapt means the agent becomes more valuable and efficient the longer it operates, continuously refining its approach based on real-world outcomes.

<img width="1456" height="1126" alt="image" src="https://github.com/user-attachments/assets/78fbba7e-09a3-4faf-a623-4772adcca6d4" />

The full article with detailed explanation: https://blog.bytebytego.com/p/what-are-ai-agents?img=https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6195b81d-b863-4721-b101-2bc8de11ffa9_1992x1366.png&open=false

# How to build Agent
I followed the instructions given in course content to create an Agent, that's why some part of code I took from provided examples. 
Hovewer, I customized chunking part and Agent egine (I prefer to run locally, so instead of OpenAI used Ollama).

Steps to build an AI Agent:
1. Data Prepearation
There is no agent if it is not fueled by data. Data might be document, readme files or FAQ's. My pioneer agent is using 'dbt-core' repo's documentation. First it should be loaded, indexing and then chunked - break large documents into smaller, manageable pieces.
This is important because:
Search relevance: Smaller chunks are more specific and relevant to user queries
Performance: AI models work better with shorter text segments
Memory limits: Large documents might exceed token limits of language models

Why We Need to Prepare Large Documents Before Using Them
Large documents create several problems:

Token limits: Most LLMs have maximum input token limits
Cost: Longer prompts cost more money
Performance: LLMs perform worse with very long contexts
Relevance: Not all parts of a long document are relevant to a specific question

Regarding chunking methods, I tried the hybrid one, which first split text to sections, section to paragraphs and then perform sliding windows ( overlap between chunks).

Code: get_data.py, chunk_data.py

2. Search Engine
To use the prepeared data, we will put it inside a search engine. This allows us to quickly find relevant information when users ask questions.
The simplest type of search is a text search. Text search works by finding all documents that contain at least one word from the query. The more words from the query that appear in a document, the more relevant that document is.

The text search types:
- lexical search - find exact matches between our query and the documents;
- vector search - uses emdedings (numerical representations of text that capture semantic meaning. Words and phrases with similar meanings have similar embeddings, even if they use different words) to identify semantically similar documents, rather than just exact word matches;
- hybrid search - combining both approaches descibed above gives us the best of both worlds.

Code: text_search.py

🔍 RAG - Retrieval-Augmented Generation.
Step 1: Retrieve
Use embeddings to get relevant chunks.
Step 2: Augment
Take those chunks and give them as context to an LLM.
Step 3: Generate
Ask the LLM to produce the answer:
- summarize
- explain
- extract data
- compare
- reason
- combine multiple retrieved parts
Without that → you just have a search engine.

❓ If embedding search already gives me the answer, why do I need an agent?
Because embeddings only retrieve text (- ask a question, - embed the question + all chunks, - find the most similar chunks, -print them)
Agents reason over retrieved text (- search once, - read chunks, - realize “I need more info”, - generate a second search query, -read more chunks, produce final answer)

3. Build an AI Agent
This an agent is an LLM that can not only generate texts, but once we give it access to our prepeared data it becomes an Agent AI.

For my pioneer agent I used Pydantic AI and Ollama 3.
   
When working with agents, the system prompt becomes one of the most essential variables we can adjust to influence our agent.
The system prompt tells the model:
- What task it should do
- How to behave
- Which tools to use and when
- What constraints exist (e.g., “Only answer from documentation”)
Without it, the model has no idea:
- if it's a chatbot
- if it should call your search_docs tool
- if it should combine chunks
- if it should refuse to answer when docs don’t contain info
- if you want JSON output
- if you want short or long answers
- if you want citations

code: agent.py

4. User Interface (to do)
It may have place to add Gradio UI to wrap a chat with Agent

# Certificate of Completion

<img width="949" height="739" alt="image" src="https://github.com/user-attachments/assets/5b2f1f2e-cee5-4c27-9081-2a282ebbf255" />















