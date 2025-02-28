# Combinatorial Alzheimer’s disease Therapeutic Efficacy Decision (Coated-LLM)
A novel framework that utilizes systematic in-context learning of large language models (LLMs) to simulate scientific reasoning and peer review to predict efficacious combinatorial therapy when data-driven prediction is infeasible.


## Workflow
<img width="1251" alt="Screenshot 2024-12-01 at 3 12 11 PM" src="https://github.com/user-attachments/assets/9edfcdef-5a33-40fd-b4bf-703c1b6c5711">

Coated-LLM is a structured framework that mimics human scientific reasoning processes to generate hypotheses on efficacious combinatorial therapy. It consists of three stages: (i) **Warm-up phase**, where *Researcher* uses external biological knowledge to practice scientific inference and keep correct predictions as learning examples. (ii) **Inference phase**, where *Researcher* inferences the new combination using its top five similar questions from learning examples and gets the consistency prediction. (iii) **Revision phase**, where multiple *Reviewers* provide feedback and *Moderator* integrates consistency prediction from *Researcher* and feedback from *Reviewers* to generate the final consensus prediction.

**Coated-LLM** is a repository for Alzheimer's disease drug combination evaluation using language models.

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/QidiXu96/Coated-LLM.git
   cd Coated-LLM-Project
2. (Optional) Create a virtual environment and activate it:
   ```bash
   python -m venv coated-test
   source coated-test/bin/activate
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run an example script:
   ```bash
   see coated-llm-example.ipynb

## Ablation Study
![figure4](https://github.com/user-attachments/assets/09a40da9-b231-4658-87fe-2d8b542465be)

• **Dynamic Few-shot**: For each target drug combination, we select the top five similar questions (based on cosine distance) from learning examples and collect corresponding reasonings.

• **External Knowledge**: We particularly focus on the pathway information (from CTDbase) that the therapeutic agents target as the external
biomedical knowledge (RAG).

• **Self-consistency**: We generate the response multiple times and aggregate them by obtaining consensus prediction via majority vote
and select the most detailed (longest) chain of thought if its paired answer is the same as the majority. 

• **Reviewer and Moderator**: We encourage multiple Reviewer LLMs to have multiple perspectives and discuss different branches of thoughts via tree
of-thoughts reasoning. Once Reviewer LLMs finish the discussion and provide feedback, the Moderator LLM aggregates the reviewer’s feedback and researcher’s response to obtain the final decision. 

## Prompt Structure
### Prompt for Researcher LLM in the Warm-up phase
```
System: You are an expert in therapy development for Alzheimer's disease and you are trying to decide if the combination of two drugs is effective or not to treat or slow the
progression of Alzheimer's disease in theory. You can identify drug targets and mechanism of action, determine biological pathways, check for multiple pathway targeting, investigate
drug-target interaction and mechanisms of synergy, consider pharmacodynamics, etc. Also, it is rare that combinations of two drugs become efficacious and synergistic in real world. As
a proficient neurobiologist, use your own knowledge and search for external information if necessary.

User: Background: <Background> {Pathway Information} </Background>. Decide if the combination of <Drug A> {Drug A Name} </Drug A> and <Drug B> {Drug B Name} </Drug B> is effective or
not to treat <Animal Model> {Animal Model Name} </Animal Model> model in theory. Take a breath and work on this problem step by step. And conclude using the format: “Effective in
theory: <Positive or Non-positive>”.
```
### Prompt for Researcher LLM in the Inference phase
```
System: You are an expert in therapy development for Alzheimer's disease and you are trying to decide if the combination of two drugs is
effective or not to treat or slow the progression of Alzheimer's disease in theory. Also, it is rare that combinations of two drugs
become efficacious and synergistic in real world. As a proficient neurobiologist, use your own knowledge and search for external
information if necessary.

User: <Question 1> {Question 1} </Question 1> : <CoT 1> {Reasoning 1} </CoT 1>
<Question 2> {Question 2} </Question 2> : <CoT 2> {Reasoning 2} </CoT 2> 
<Question 3> {Question 3} </Question 3> : <CoT 3> {Reasoning 3} </CoT 3> 
<Question 4> {Question 4} </Question 4> : <CoT 4> {Reasoning 4} </CoT 4> 
<Question 5> {Question 5} </Question 5> : <CoT 5> {Reasoning 5} </CoT 5> 
<Background> {Pathway Information} </Background >
<Test Question> {Test Question} </Test Question>
Take a breath and work on this problem step by step. And conclude using the format 'Effective in theory: <Positive or Non-positive>.'
```

### Prompt for Reviewer LLM in the Revision (evaluate) phase
```
System: Imagine three different experts who are in therapy development for Alzheimer's disease, are tasked with critically reviewing the
reasoning and conclusions regarding the effectiveness of a combination of two drugs on an Alzheimer's disease animal model from a
theoretical perspective. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on
to the next step, etc. If any expert realizes they're wrong at any point then they leave. At the end of the discussion, the remaining
experts will summarize their conclusions, highlighting any potential drug interactions that could limit or enhance effectiveness.

User: Previous response: <Response> {Selected most detailed reasoning from the Inference phase} </Response>. Please evaluate the
response. Explore the potential for drug interactions that could limit or enhance effectiveness.
```

### Prompt for Moderator LLM in the Revision (revise) phase
```
System: You are an expert in therapy development for Alzheimer's disease and you are trying to decide if the combination of two drugs is
effective or not to treat or slow the progression of Alzheimer's disease in theory. Also, it is rare that combinations of two drugs
become efficacious and synergistic in real world. As a proficient neurobiologist, use your own knowledge and search for external information if
necessary.

User: Previous response: <Response> {Selected most detailed reasoning from the Inference phase} </Response >
Feedback: <Feedback> {Detailed Feedback} </Feedback >
Based on the previous response and feedback, <Question> {Test Question} </Question >. Take a breath and work on this problem step by
step. And conclude using the format 'Effective in theory: <Positive or Non-positive>.'
```
