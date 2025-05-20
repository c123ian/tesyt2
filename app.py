from fasthtml.common import *
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import json
import uuid
import datetime
import traceback
import re

# Initialize FastHTML app with Tailwind and DaisyUI
app, rt = fast_app()

# Set your API key

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Try the specific model name
    google_api_key="AIzxxxxxxxxx"  # Pass key directly
)

# MRCPCH TAS Guidelines
MRCPCH_GUIDELINES = """
- All questions must include a clinical vignette with age and sex of patient listed (2-4 lines ideal)
- Investigations should be presented in a table with normal values
- Five plausible answer options (A-E) with only one correct answer
- Question must use "most likely" or "most appropriate" phrasing
- "Select one answer only" instruction must be present
- Correct answer explanation should be 4-6 lines
- Incorrect answer explanations should be 2-3 lines each
- All explanations must be educational and unique
- Use UK medical terminology and spelling
"""

# Pastest House Style Guidelines
PASTEST_STYLE_GUIDELINES = """
- Use UK spelling (not US) - avoid Americanisms like "-ize"; use "-ise" instead
- Use gender-neutral language (except in case studies)
- Single spacing between sentences (not double)
- Format syndromes WITHOUT apostrophe 's' (e.g., "Down syndrome")
- Format diseases WITH apostrophe 's' (e.g., "Alzheimer's disease")
- Use proper age referents: newborns/neonates (0-1 month), infants (1-24 months), children (2-13 years), adolescents (13-17 years), adults (18+ years)
- Spell out abbreviations the first time with abbreviation in parentheses (except for approved abbreviations)
- Numbers 1-10 should be spelled out unless with units of measurement
- Use proper spacing between numbers and units (e.g., "5 kg" not "5kg")
- Use SI units when appropriate
- Tables should have first line in bold
"""

# Approved Abbreviations List
APPROVED_ABBREVIATIONS = [
    "AF", "BMI", "BP", "CNS", "CSF", "CT", "DNA", "DNR", "DNAR", "DNACPR",
    "DVT", "ECG", "ECHO", "EEG", "FBC", "GCS", "GI", "HbA1c", "HIV", "HRT",
    "IM", "IV", "LFT", "LFTs", "MRI", "MRSA", "NG", "NSAIDs", "PO", "PCR",
    "PRN", "RNA", "RTA", "STEMI", "NSTEMI", "UTI", "UK", "U&E", "U&Es", "X-ray", "4AT"
]

# Type definitions
class MCPCHQuestion(TypedDict):
    vignette: str
    patient_details: dict
    investigations: List[dict]
    question: str
    options: List[dict]
    explanations: List[dict]

class QuestionState(TypedDict):
    question: MCPCHQuestion
    results: Dict[str, Dict[str, bool]]
    corrections: List[str]
    score: float
    rewritten_question: MCPCHQuestion

# Helper function to find undefined abbreviations
def find_undefined_abbreviations(text, approved_list=APPROVED_ABBREVIATIONS):
    """Find abbreviations in text that aren't in the approved list and might be undefined"""
    # Simple regex to find potential abbreviations (2+ uppercase letters)
    potential_abbrs = re.findall(r'\b[A-Z]{2,}(?:\d*[A-Z]*)*\b', text)
    
    # Filter out those in the approved list
    unapproved = [abbr for abbr in potential_abbrs if abbr not in approved_list]
    
    # Return unique unapproved abbreviations
    return list(set(unapproved))

# Helper function to extract all text from a question
def extract_all_text(question):
    """Extracts all text from a question object for abbreviation checking"""
    texts = []
    
    # Extract vignette
    if "vignette" in question:
        texts.append(question["vignette"])
    
    # Extract question
    if "question" in question:
        texts.append(question["question"])
    
    # Extract options
    if "options" in question:
        for option in question["options"]:
            if "text" in option:
                texts.append(option["text"])
    
    # Extract explanations
    if "explanations" in question:
        for explanation in question["explanations"]:
            if "text" in explanation:
                texts.append(explanation["text"])
    
    return " ".join(texts)

# Evaluation functions
def check_format(state: QuestionState) -> QuestionState:
    """Check all format-related requirements"""
    question = state["question"]
    
    prompt = f"""
    Evaluate this MRCPCH TAS exam question for FORMAT compliance with official guidelines.
    For each requirement, respond with PASS or FAIL followed by specific feedback:

    1. Vignette length (2-4 lines ideal)
    2. Patient demographics (age and sex specified)
    3. Investigation table format (with normal ranges in right column)
    4. Five plausible answer options (A-E)
    5. Question uses "most likely" or "most appropriate" phrasing
    6. "Select one answer only" instruction present

    Question:
    {json.dumps(question, indent=2)}
    
    Return your evaluation as a valid JSON object with this exact format:
    {{
      "vignette_length": {{"passed": true, "feedback": "explanation"}},
      "patient_demographics": {{"passed": true, "feedback": "explanation"}},
      "investigation_table": {{"passed": true, "feedback": "explanation"}},
      "five_options": {{"passed": true, "feedback": "explanation"}},
      "question_phrasing": {{"passed": true, "feedback": "explanation"}},
      "select_instruction": {{"passed": true, "feedback": "explanation"}}
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from the response
        result = {}
        try:
            # Try to parse the whole response as JSON
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                result = json.loads(json_str)
            else:
                # Default results if JSON extraction fails
                result = {
                    "vignette_length": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "patient_demographics": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "investigation_table": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "five_options": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "question_phrasing": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "select_instruction": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"}
                }
        
        # Add results to state
        state["results"]["format"] = result
        
        # Extract corrections needed
        for check, check_result in result.items():
            if not check_result.get("passed", True):
                state["corrections"].append(check_result.get("feedback", "Issue with " + check))
        
        return state
    except Exception as e:
        # Handle errors gracefully
        state["results"]["format"] = {
            "error": {"passed": False, "feedback": f"Error in format check: {str(e)}"}
        }
        state["corrections"].append(f"Error in format check: {str(e)}")
        return state

def check_content(state: QuestionState) -> QuestionState:
    """Check all content-related requirements"""
    question = state["question"]
    
    prompt = f"""
    Evaluate this MRCPCH TAS exam question for CONTENT quality with official guidelines.
    For each requirement, respond with PASS or FAIL followed by specific feedback:

    1. Medical accuracy and relevance to pediatrics
    2. Question tests interpretation rather than isolated facts
    3. All five options are plausible (no obviously incorrect options)
    4. Only one clearly correct answer

    Question:
    {json.dumps(question, indent=2)}
    
    Return your evaluation as a valid JSON object with this exact format:
    {{
      "medical_accuracy": {{"passed": true, "feedback": "explanation"}},
      "tests_interpretation": {{"passed": true, "feedback": "explanation"}},
      "plausible_options": {{"passed": true, "feedback": "explanation"}},
      "single_correct_answer": {{"passed": true, "feedback": "explanation"}}
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from the response
        result = {}
        try:
            # Try to parse the whole response as JSON
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                result = json.loads(json_str)
            else:
                # Default results if JSON extraction fails
                result = {
                    "medical_accuracy": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "tests_interpretation": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "plausible_options": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "single_correct_answer": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"}
                }
        
        # Add results to state
        state["results"]["content"] = result
        
        # Extract corrections needed
        for check, check_result in result.items():
            if not check_result.get("passed", True):
                state["corrections"].append(check_result.get("feedback", "Issue with " + check))
        
        return state
    except Exception as e:
        # Handle errors gracefully
        state["results"]["content"] = {
            "error": {"passed": False, "feedback": f"Error in content check: {str(e)}"}
        }
        state["corrections"].append(f"Error in content check: {str(e)}")
        return state

def check_explanations(state: QuestionState) -> QuestionState:
    """Check all explanation-related requirements"""
    question = state["question"]
    
    prompt = f"""
    Evaluate this MRCPCH TAS exam question for EXPLANATION quality with official guidelines.
    For each requirement, respond with PASS or FAIL followed by specific feedback:

    1. Correct answer explanation length (4-6 lines)
    2. Incorrect answer explanations length (2-3 lines each)
    3. Explanations are educational (provide relevant information)
    4. Each explanation is unique (no repetition)
    5. UK medical terminology and spelling

    Question:
    {json.dumps(question, indent=2)}
    
    Return your evaluation as a valid JSON object with this exact format:
    {{
      "correct_explanation_length": {{"passed": true, "feedback": "explanation"}},
      "incorrect_explanation_length": {{"passed": true, "feedback": "explanation"}},
      "educational_value": {{"passed": true, "feedback": "explanation"}},
      "unique_explanations": {{"passed": true, "feedback": "explanation"}},
      "uk_terminology": {{"passed": true, "feedback": "explanation"}}
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from the response
        result = {}
        try:
            # Try to parse the whole response as JSON
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                result = json.loads(json_str)
            else:
                # Default results if JSON extraction fails
                result = {
                    "correct_explanation_length": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "incorrect_explanation_length": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "educational_value": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "unique_explanations": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "uk_terminology": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"}
                }
        
        # Add results to state
        state["results"]["explanations"] = result
        
        # Extract corrections needed
        for check, check_result in result.items():
            if not check_result.get("passed", True):
                state["corrections"].append(check_result.get("feedback", "Issue with " + check))
        
        return state
    except Exception as e:
        # Handle errors gracefully
        state["results"]["explanations"] = {
            "error": {"passed": False, "feedback": f"Error in explanations check: {str(e)}"}
        }
        state["corrections"].append(f"Error in explanations check: {str(e)}")
        return state

def check_style(state: QuestionState) -> QuestionState:
    """Check compliance with Pastest House Style Guide"""
    question = state["question"]
    
    # Find any potentially undefined abbreviations
    all_text = extract_all_text(question)
    undefined_abbrs = find_undefined_abbreviations(all_text)
    undefined_abbrs_str = ", ".join(undefined_abbrs) if undefined_abbrs else "None found"
    
    # Convert approved abbreviations list to a string for the prompt
    approved_abbr_str = ", ".join(APPROVED_ABBREVIATIONS)
    
    prompt = f"""
    Evaluate this MRCPCH TAS exam question for STYLE compliance with Pastest House Style Guide.
    For each requirement, respond with PASS or FAIL followed by specific feedback:

    1. UK spelling and terminology (not US spelling, e.g., "-ise" not "-ize")
    2. Gender-neutral language (except in specific case descriptions)
    3. Medical terminology formatting:
       - Syndromes without apostrophe 's' (e.g., "Down syndrome")
       - Diseases with apostrophe 's' (e.g., "Alzheimer's disease")
       - Correct age referents (newborn 0-1 month, infant 1-24 months, etc.)
    4. Abbreviations:
       - Only use approved abbreviations without definition, or define others on first use
       - Approved abbreviations: {approved_abbr_str}
       - Non-approved abbreviations detected: {undefined_abbrs_str}
       - Check if these non-approved abbreviations are spelled out first time with abbreviation in parentheses
    5. Number and unit formatting:
       - Numbers 1-10 spelled out unless with units
       - Proper spacing between numbers and units (e.g., "5 kg" not "5kg")
       - Correct SI units where appropriate
    6. Text formatting:
       - Single spacing between sentences (not double)
       - Tables formatted correctly with first line in bold

    Question:
    {json.dumps(question, indent=2)}
    
    Return your evaluation as a valid JSON object with this exact format:
    {{
      "uk_spelling": {{"passed": true, "feedback": "explanation"}},
      "gender_neutral": {{"passed": true, "feedback": "explanation"}},
      "medical_terminology": {{"passed": true, "feedback": "explanation"}},
      "abbreviations": {{"passed": true, "feedback": "explanation"}},
      "number_formatting": {{"passed": true, "feedback": "explanation"}},
      "text_formatting": {{"passed": true, "feedback": "explanation"}}
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from the response
        result = {}
        try:
            # Try to parse the whole response as JSON
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                result = json.loads(json_str)
            else:
                # Default results if JSON extraction fails
                result = {
                    "uk_spelling": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "gender_neutral": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "medical_terminology": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "abbreviations": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "number_formatting": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"},
                    "text_formatting": {"passed": False, "feedback": "Failed to evaluate - LLM response parsing error"}
                }
        
        # Add results to state
        state["results"]["style"] = result
        
        # Extract corrections needed
        for check, check_result in result.items():
            if not check_result.get("passed", True):
                state["corrections"].append(check_result.get("feedback", "Issue with " + check))
        
        return state
    except Exception as e:
        # Handle errors gracefully
        state["results"]["style"] = {
            "error": {"passed": False, "feedback": f"Error in style check: {str(e)}"}
        }
        state["corrections"].append(f"Error in style check: {str(e)}")
        return state

def calculate_score(state: QuestionState) -> QuestionState:
    """Calculate normalized score (0-1) based on all checks"""
    results = state["results"]
    total_checks = 0
    passed_checks = 0
    
    for category, checks in results.items():
        for check, result in checks.items():
            if check == "error":
                continue
            total_checks += 1
            if result.get("passed", False):
                passed_checks += 1
    
    score = passed_checks / total_checks if total_checks > 0 else 0
    state["score"] = round(score, 2)
    return state

def rewrite_if_needed(state: QuestionState) -> QuestionState:
    """If corrections are needed, rewrite the question"""
    if not state["corrections"]:
        state["rewritten_question"] = None
        return state
    
    question = state["question"]
    corrections = state["corrections"]
    
    # Find any potentially undefined abbreviations to provide specific guidance
    all_text = extract_all_text(question)
    undefined_abbrs = find_undefined_abbreviations(all_text)
    undefined_abbrs_str = ", ".join(undefined_abbrs) if undefined_abbrs else "None found"
    
    corrections_text = "\n".join([f"- {correction}" for correction in corrections])
    
    # Convert approved abbreviations list to a string for the prompt
    approved_abbr_str = ", ".join(APPROVED_ABBREVIATIONS)
    
    prompt = f"""
    Rewrite this MRCPCH TAS exam question to fix ONLY these specific issues:
    {corrections_text}
    
    Original question:
    {json.dumps(question, indent=2)}
    
    Maintain the medical accuracy and core testing concept.
    Preserve the same correct answer if possible.
    
    Follow these style guidelines:
    - Use UK medical terminology and spelling (not US)
    - Use "-ise" not "-ize" endings
    - Use gender-neutral language (except in specific case descriptions)
    - Format syndromes WITHOUT apostrophe 's' (e.g., "Down syndrome")
    - Format diseases WITH apostrophe 's' (e.g., "Alzheimer's disease")
    - Use proper age referents: 
      * newborns/neonates (0-1 month)
      * infants (1-24 months)
      * children (2-13 years)
      * adolescents (13-17 years)
      * adults (18+ years)
    - For abbreviations:
      * Approved abbreviations that can be used without definition: {approved_abbr_str}
      * All other abbreviations must be spelled out on first use with the abbreviation in parentheses
      * Non-approved abbreviations detected that may need defining: {undefined_abbrs_str}
    - Format numbers 1-10 as words unless with units of measurement
    - Use proper spacing between numbers and units (e.g., "5 kg" not "5kg")
    - Use SI units when appropriate
    - Use single spacing between sentences
    - Format tables with the first line in bold
    
    Return your rewritten question as a valid JSON object with the same structure as the original.
    """
    
    try:
        response = llm.invoke(prompt)
        
        # Extract the JSON object from the response
        try:
            # Try to parse the whole response as JSON
            rewritten_question = json.loads(response.content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = content[start:end]
                rewritten_question = json.loads(json_str)
            else:
                # Default if JSON extraction fails
                rewritten_question = {"error": "Failed to parse rewritten question", "raw_response": response.content}
        
        state["rewritten_question"] = rewritten_question
    except Exception as e:
        # Fallback if parsing fails
        state["rewritten_question"] = {"error": f"Failed to rewrite question: {str(e)}", "raw_response": "Error occurred"}
    
    return state

# Create the evaluation workflow
def create_question_graph():
    workflow = StateGraph(QuestionState)
    
    # Add nodes
    workflow.add_node("check_format", check_format)
    workflow.add_node("check_content", check_content)
    workflow.add_node("check_explanations", check_explanations)
    workflow.add_node("check_style", check_style)
    workflow.add_node("calculate_score", calculate_score)
    workflow.add_node("rewrite_if_needed", rewrite_if_needed)
    
    # Define edges
    workflow.set_entry_point("check_format")
    workflow.add_edge("check_format", "check_content")
    workflow.add_edge("check_content", "check_explanations")
    workflow.add_edge("check_explanations", "check_style")
    workflow.add_edge("check_style", "calculate_score")
    workflow.add_edge("calculate_score", "rewrite_if_needed")
    workflow.add_edge("rewrite_if_needed", END)
    
    return workflow.compile()

def evaluate_and_rewrite(question):
    """Main function to evaluate and possibly rewrite a question"""
    try:
        graph = create_question_graph()
        
        # Initialize state
        state = {
            "question": question,
            "results": {},
            "corrections": [],
            "score": 0.0,
            "rewritten_question": None
        }
        
        # Run the evaluation graph
        result = graph.invoke(state)
        return result
    except Exception as e:
        # Return error info if graph fails
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "question": question,
            "results": {},
            "corrections": [f"System error: {str(e)}"],
            "score": 0.0,
            "rewritten_question": None
        }

# Database simulation (would be replaced with actual DB in production)
question_db = {}

@rt("/")
def get():
    # Main content
    content = Div(
        H1("MRCPCH TAS Question Evaluator", cls="text-4xl font-bold mb-6 text-center text-primary"),
        
        # Guidelines section with collapsible elements
        Div(
            H2("Guidelines", cls="text-2xl font-semibold mb-4"),
            
            # MRCPCH Guidelines (collapsible)
            Details(
                Summary("MRCPCH TAS Guidelines", cls="text-xl font-semibold cursor-pointer p-4 hover:bg-white hover:shadow-sm rounded-lg"),
                Div(
                    Pre(MRCPCH_GUIDELINES, cls="p-4 bg-white border border-base-300 rounded-lg text-sm"),
                    cls="mt-2 p-4"
                ),
                cls="mb-4 border border-base-300 rounded-lg"
            ),
            
            # Pastest Style Guidelines (collapsible)
            Details(
                Summary("Pastest House Style Guidelines", cls="text-xl font-semibold cursor-pointer p-4 hover:bg-white hover:shadow-sm rounded-lg"),
                Div(
                    Pre(PASTEST_STYLE_GUIDELINES, cls="p-4 bg-white border border-base-300 rounded-lg text-sm"),
                    cls="mt-2 p-4"
                ),
                cls="mb-4 border border-base-300 rounded-lg"
            ),
            
            # Approved Abbreviations (collapsible)
            Details(
                Summary("Approved Abbreviations", cls="text-xl font-semibold cursor-pointer p-4 hover:bg-white hover:shadow-sm rounded-lg"),
                Div(
                    P("The following medical abbreviations can be used without definition:", cls="mb-2"),
                    Div(
                        ", ".join(APPROVED_ABBREVIATIONS),
                        cls="p-4 bg-white border border-base-300 rounded-lg text-sm"
                    ),
                    cls="mt-2 p-4"
                ),
                cls="mb-4 border border-base-300 rounded-lg"
            ),
            
            cls="bg-white p-6 rounded-lg shadow-md mb-6"
        ),
        
        # Question submission section
        Div(
            H2("Submit a Question for Evaluation", cls="text-2xl font-semibold mb-4"),
            P("Paste your question in JSON format or use the sample provided.", cls="mb-4"),
            Form(
                Div(
                    Label("Question JSON:", cls="font-semibold"),
                    Textarea(
                        id="question_json", 
                        name="question_json", 
                        value="{\n  \"vignette\": \"A 3-year-old boy presents with fever.\",\n  \"patient_details\": {\n    \"age\": 3,\n    \"sex\": \"male\"\n  },\n  \"investigations\": [\n    {\"test\": \"WBC\", \"value\": \"15 x 10^9/L\", \"normal_range\": \"5-15 x 10^9/L\"},\n    {\"test\": \"CRP\", \"value\": \"50 mg/L\", \"normal_range\": \"<10 mg/L\"}\n  ],\n  \"question\": \"What is the most likely diagnosis?\",\n  \"options\": [\n    {\"letter\": \"A\", \"text\": \"Viral URTI\", \"correct\": true},\n    {\"letter\": \"B\", \"text\": \"Bacterial pneumonia\", \"correct\": false},\n    {\"letter\": \"C\", \"text\": \"Meningitis\", \"correct\": false},\n    {\"letter\": \"D\", \"text\": \"Otitis media\", \"correct\": false},\n    {\"letter\": \"E\", \"text\": \"Urinary tract infection\", \"correct\": false}\n  ],\n  \"explanations\": [\n    {\"letter\": \"A\", \"text\": \"This is the correct answer. Viral URTIs commonly present with fever in children.\"},\n    {\"letter\": \"B\", \"text\": \"Pneumonia would typically include respiratory symptoms.\"},\n    {\"letter\": \"C\", \"text\": \"Meningitis would include more severe symptoms.\"},\n    {\"letter\": \"D\", \"text\": \"Otitis media would include ear pain.\"},\n    {\"letter\": \"E\", \"text\": \"UTI would include urinary symptoms.\"}\n  ]\n}",
                        cls="w-full p-2 border rounded mb-4 font-mono text-sm",
                        rows="15"
                    ),
                    cls="mb-4"
                ),
                Button(
                    "Evaluate Question", 
                    cls="btn btn-primary w-full", 
                    hx_post="/evaluate-question", 
                    hx_target="#evaluation_results", 
                    hx_indicator="#backdrop"  # Point to the backdrop
                ),
                cls="bg-white p-6 rounded-lg shadow-md"
            ),
            cls="mb-6"
        ),
        
        # Backdrop with spinner (fixed positioning)
        Div(
            id="backdrop",
            cls="fixed top-0 bottom-0 left-0 right-0 bg-white bg-opacity-70 opacity-0 -z-10 transition-all duration-300 flex items-center justify-center",
            children=[
                # Custom spinner using div instead of relying on DaisyUI's loading class
                Div(cls="custom-spinner")
            ]
        ),
        
        # Results section
        Div(id="evaluation_results", cls="mt-4"),
        
        cls="main-container"
    )
    
    # Return the title, styles, and content separately
    return (
        Title("MRCPCH TAS Question Evaluator"), 
        Link(href="https://cdn.jsdelivr.net/npm/daisyui@5", rel="stylesheet", type="text/css"), 
        Script(src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"), 
        # Add Cascadia Code font
        Link(rel="preconnect", href="https://fonts.googleapis.com"),
        Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        Link(href="https://fonts.googleapis.com/css2?family=Cascadia+Code:ital,wght@0,200..700;1,200..700&display=swap", rel="stylesheet"),
        Style("""
            :root {
                --color-primary: #02b383;
                --color-primary-focus: #019d71;
                --color-primary-content: #f9f9fa;
                --color-secondary: #e53839;
                --color-secondary-focus: #d32f30;
                --color-secondary-content: #f9f9fa;
                --color-accent: #02b383;
                --color-accent-focus: #019d71;
                --color-accent-content: #f9f9fa;
                --color-neutral: #f9f9fa;
                --color-neutral-focus: #e9e9eb;
                --color-neutral-content: #333333;
                --color-base-100: #ffffff;
                --color-base-200: #ffffff;
                --color-base-300: #f0f0f0;
                --color-base-content: #333333;
                --color-info: #3abff8;
                --color-success: #02b383;
                --color-warning: #fbbd23;
                --color-error: #e53839;
            }
            
            body {
                background-color: #ffffff;
            }
            
            body, h1, h2, h3, h4, h5, h6, p, span, div, button, input, textarea, label, summary, details {
                font-family: 'Cascadia Code', monospace;
            }
            
            .main-container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 1rem;
            }
            
            pre {
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Cascadia Code', monospace;
                background-color: #ffffff;
            }
            
            details {
                transition: all 0.3s ease;
                background-color: #ffffff;
            }
            
            summary {
                list-style: none;
                background-color: #ffffff;
            }
            
            summary::-webkit-details-marker {
                display: none;
            }
            
            summary::after {
                content: '+';
                float: right;
            }
            
            details[open] summary::after {
                content: '-';
            }
            
            .btn-primary {
                background-color: var(--color-primary);
                color: var(--color-primary-content);
            }
            
            .btn-primary:hover {
                background-color: var(--color-primary-focus);
            }
            
            .badge-success {
                background-color: var(--color-success);
                color: var(--color-primary-content);
            }
            
            .badge-error {
                background-color: var(--color-error);
                color: var(--color-primary-content);
            }
            
            .text-primary {
                color: var(--color-primary);
            }
            
            .text-error {
                color: var(--color-error);
            }
            
            /* Backdrop and spinner */
            #backdrop {
                transition: opacity 0.3s, z-index 0s 0.3s;
                backdrop-filter: blur(2px);
            }
            
            #backdrop.htmx-request {
                opacity: 0.85;
                z-index: 50;
                transition: opacity 0.3s, z-index 0s;
            }
            
            /* Explicit custom spinner that doesn't rely on DaisyUI */
            .custom-spinner {
                width: 48px;
                height: 48px;
                border-radius: 50%;
                border: 6px solid #ffffff;
                border-color: var(--color-primary) transparent var(--color-primary) transparent;
                animation: spinner 1.2s linear infinite;
            }
            
            @keyframes spinner {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        """),
        content
    )

@rt("/evaluate-question")
async def post(question_json: str):
    try:
        # Parse the question JSON
        question = json.loads(question_json)
        
        # Generate a question ID if not present
        question_id = question.get("id", str(uuid.uuid4()))
        question["id"] = question_id
        
        # Perform evaluation and rewriting
        result = evaluate_and_rewrite(question)
        
        # Store in our simulated database
        question_db[question_id] = {
            "original_question": question,
            "evaluation": result,
            "evaluated_at": datetime.datetime.now().isoformat()
        }
        
        # Check if any failures
        has_failures = len(result.get("corrections", [])) > 0
        
        # Calculate score
        score = result.get("score", 0)
        score_percent = int(score * 100)
        
        # Format the results for display
        results_div = Div(
            H2("Evaluation Results", cls="text-2xl font-semibold mb-4"),
            
            # Error message if present
            Div(
                H3("Error", cls="text-xl font-semibold mb-4 text-error"),
                P(result.get("error", ""), cls="mb-4"),
                Pre(result.get("traceback", ""), cls="bg-base-200 p-4 rounded-lg text-sm"),
                cls="bg-white p-6 rounded-lg shadow-md mb-6"
            ) if "error" in result else "",
            
            # Score and overall status
            Div(
                Div(
                    Div(
                        Span("Quality Score: ", cls="font-bold text-lg"),
                        Div(f"{score_percent}%", 
                            cls=f"badge ml-2 {'badge-success' if score >= 0.8 else 'badge-warning' if score >= 0.6 else 'badge-error'}"),
                        cls="mb-2"
                    ),
                    Div(
                        Span("Status: ", cls="font-bold text-lg"),
                        Div("Passed", cls="badge badge-success ml-2") if not has_failures else 
                        Div("Needs Revision", cls="badge badge-error ml-2"),
                        cls="mb-2"
                    ),
                    cls="flex flex-col md:flex-row md:justify-between"
                ),
                cls="p-4 bg-base-200 rounded-box mb-6"
            ),
            
            # Category results with collapsible details
            Div(
                H3("Detailed Assessment", cls="text-xl font-semibold mb-4"),
                
                # Format checks
                details_for_category("Format", result.get("results", {}).get("format", {})),
                
                # Content checks
                details_for_category("Content", result.get("results", {}).get("content", {})),
                
                # Explanation checks
                details_for_category("Explanations", result.get("results", {}).get("explanations", {})),
                
                # Style checks
                details_for_category("Style", result.get("results", {}).get("style", {})),
                
                cls="mb-6 bg-white p-6 rounded-lg shadow-md"
            ),
            
            # Issues requiring correction
            Div(
                H3("Issues Requiring Correction", cls="text-xl font-semibold mb-4"),
                P("No issues found! This question meets all guidelines.", cls="text-success") 
                if not result.get("corrections", []) else
                Div(
                    Ul(*[Li(correction, cls="mb-2") for correction in result.get("corrections", [])], 
                       cls="list-disc pl-5"),
                    cls="p-4 bg-base-200 rounded-box"
                ),
                cls="mb-6 bg-white p-6 rounded-lg shadow-md"
            ),
            
            # Rewritten question (if available)
            Div(
                H3("Rewritten Question", cls="text-xl font-semibold mb-4"),
                P("No rewriting needed. The original question meets all guidelines.", cls="text-success") 
                if not result.get("rewritten_question") else
                Div(
                    Pre(
                        json.dumps(result.get("rewritten_question", {}), indent=2), 
                        cls="bg-base-200 p-4 rounded-lg overflow-auto text-sm"
                    ),
                    cls="mt-4"
                ),
                cls="bg-white p-6 rounded-lg shadow-md"
            ),
            
            cls="bg-white p-6 rounded-lg shadow-md"
        )
        
        return results_div
        
    except Exception as e:
        # Handle errors
        return Div(
            H2("Error Processing Question", cls="text-2xl font-semibold mb-4 text-error"),
            P(f"An error occurred: {str(e)}", cls="mb-4"),
            Pre(traceback.format_exc(), cls="bg-base-200 p-4 rounded-lg text-sm"),
            cls="bg-white p-6 rounded-lg shadow-md"
        )

def details_for_category(category_name, checks):
    """Helper function to create a details/summary section for a category of checks"""
    # Filter out any error entries
    valid_checks = {k: v for k, v in checks.items() if k != "error"}
    
    # Count passes and failures
    total = len(valid_checks)
    passes = sum(1 for check in valid_checks.values() if check.get("passed", False))
    
    # Handle error case
    if "error" in checks:
        return Details(
            Summary(
                f"{category_name} Checks: Error", 
                cls="font-semibold cursor-pointer p-4 hover:bg-base-200 rounded-t-lg"
            ),
            Div(
                P(checks["error"].get("feedback", "Unknown error"), cls="text-error"),
                cls="p-4"
            ),
            cls="mb-2 border border-base-300 rounded-lg"
        )
    
    # Normal case
    return Details(
        Summary(
            f"{category_name} Checks: {passes}/{total} Passed", 
            cls="font-semibold cursor-pointer p-4 hover:bg-base-200 rounded-t-lg"
        ),
        Div(
            *[
                Div(
                    Div(
                        Span(check_name.replace("_", " ").title(), cls="font-medium"),
                        Div(
                            "Pass", 
                            cls="badge badge-success ml-2" if check_result.get("passed", False) else "badge badge-error ml-2"
                        ),
                        cls="flex items-center justify-between"
                    ),
                    P(check_result.get("feedback", ""), cls="mt-1 text-sm"),
                    cls="p-3 bg-base-200 rounded-md mb-2"
                )
                for check_name, check_result in valid_checks.items()
            ],
            cls="p-4"
        ),
        cls="mb-2 border border-base-300 rounded-lg"
    )

serve()
