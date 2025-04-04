import re
import json
import google.generativeai as genai  # Correct import for Gemini
import text_parser  # Your existing rule-based parser module

# Set your Gemini API key (ensure that this key is kept secure)
genai.configure(api_key="AIzaSyA3ronSnLTaKyLEX7sgG4LSm-EEDYZ1DkI")

def extract_logic_forms_from_llm(llm_response: str):
    """
    Extracts logic forms (substrings enclosed in square brackets) from the LLM's response.
    """
    forms = re.findall(r'\[.+?\]', llm_response)
    forms = [form.strip() for form in forms]
    # Remove duplicates while preserving order
    seen = set()
    unique_forms = []
    for form in forms:
        if form not in seen:
            unique_forms.append(form)
            seen.add(form)
    return unique_forms

def llm_parse(text: str):
    """
    Uses the Gemini API to convert the geometry problem text into formal logic forms.
    
    The prompt instructs the model to extract all formal logic forms (enclosed in square brackets)
    that represent geometric objects or relations. The output should be a list of logic forms with
    no additional commentary.
    """
    prompt = (
        "You are an advanced geometry parser. Your task is to read the given geometry problem text "
        "and extract all formal logic forms from it. A logic form is a concise representation of geometric "
        "objects or relations enclosed in square brackets, such as [Triangle(A,B,C)], [Angle(ABC)], or [Parallel(Line(A,B),Line(C,D))].\n\n"
        "Please analyze the text carefully, infer any implicit geometric relations, and output only the list "
        "of logic forms in a comma-separated format. Do not include any explanation or commentary.\n\n"
        "Geometry Problem Text:\n"
        f"{text}\n\n"
        "Output (e.g., [Triangle(A,B,C)], [Angle(ABC)], ...):"
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')  # Use the appropriate Gemini model name
        response = model.generate_content(prompt)
        llm_output = response.text
        llm_logic_forms = extract_logic_forms_from_llm(llm_output)
    except Exception as e:
        print("Error calling Gemini API:", e)
        llm_logic_forms = []
        llm_output = ""
        
    return llm_logic_forms, llm_output

def hybrid_parse(text: str):
    """
    Hybrid parser that combines the output from the existing rule-based parser and the Gemini-based parser.
    
    1. It first obtains the logic forms from your rule-based parser (via text_parser.parse).
    2. It then obtains the logic forms from the Gemini API.
    3. Finally, it merges the two sets of logic forms (taking the union) and returns the combined output.
    """
    # Get the rule-based parser output
    rule_logic_forms, rule_output_text, rule_reduced_text = text_parser.parse(text)
    
    # Get the Gemini-based parser output
    llm_logic_forms, llm_output_text = llm_parse(text)
    
    # Combine the logic forms, removing duplicates while preserving order
    combined_logic_forms = list(dict.fromkeys(rule_logic_forms + llm_logic_forms))
    
    return {
        "combined_logic_forms": combined_logic_forms,
        "rule_based_output": rule_output_text,
        "llm_output": llm_output_text,
        "reduced_text": rule_reduced_text
    }

if __name__ == "__main__":
    # Example geometry problem text
    example_text = (
        "In triangle ABC, side AB is 5, side BC is 7, and angle ABC is 60 degrees. "
        "Find the area of triangle ABC."
    )
    
    results = hybrid_parse(example_text)
    
    print("=== Rule-Based Parser Output ===")
    print(results["rule_based_output"])
    print("\n=== Gemini LLM Parser Output ===")
    print(results["llm_output"])
    print("\n=== Combined Logic Forms ===")
    for form in results["combined_logic_forms"]:
        print(form)