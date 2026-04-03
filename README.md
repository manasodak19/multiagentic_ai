Multi-Agent Systems (MAS) Gallery
This repository is a comprehensive showcase of diverse architectural patterns and coordination strategies for Large Language Models. It serves as a practical guide for orchestrating multiple AI agents to solve complex document intelligence and reasoning tasks.

📂 Project Portfolio
1. Document Intelligence Hub (intelligence-hub)
Design Pattern: Parallel Execution & Adaptive Retry
This system transforms PDFs into structured reports using a 7-agent swarm.

Workflow: A Planner coordinates four specialized agents—Summarizer, Fact Extractor, Quiz Generator, and Gap Analyzer—which execute simultaneously.

Quality Control: A Critic agent grades the output. If the quality score is low, the system triggers an Adaptive Retry, feeding the Critic’s feedback back to the Planner for a refined second pass.

2. Multi-Agent Debate System (debate-system)
Design Pattern: Adversarial Architecture
A sophisticated 7-agent framework designed to conduct structured debates based on document evidence.

Workflow: Pairs of Researchers and Debaters represent "FOR" and "AGAINST" sides. A Judge evaluates the performance across five criteria.

Dynamic Logic: If the decision is a "razor-thin" margin, a Cross-Examination round is triggered where agents identify weaknesses in the opponent's arguments before a Synthesizer generates a final conclusion.

3. Document QA System (qa-system)
Design Pattern: Sequential Pipeline with Fact-Checking
A lightweight, rate-limit-friendly project focused on reliable linear pipelines.

Workflow: A three-agent chain consisting of a Planner for search query generation, an Answerer to synthesize information, and a Verifier to automate fact-checking.

Milestones: Includes implementations for confidence-based retries and multi-question processing.

4. CrewAI Examples (crewai)
Design Pattern: Role-Based Collaboration
A collection of implementations using the CrewAI framework, demonstrating high-level abstractions where agents are assigned specific "Roles," "Goals," and "Tools" to collaborate autonomously on tasks like data analysis.

🛠️ Tech Stack
Orchestration: Python, CrewAI, Custom Agent Logic

Web Interface: Streamlit

Document Processing: PyPDF2, LangChain
