#!/usr/bin/env python
# coding: utf-8

import json
import ast
import time
from tqdm import tqdm

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

def clean_and_parse_response(response_text):
    """
    Strips markdown formatting (e.g. triple backticks) and safely parses into Python list.
    """
    response_text = response_text.strip()
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = "\n".join(response_text.splitlines()[1:-1])

    try:
        return ast.literal_eval(response_text)
    except Exception:
        return []

def evaluate(diagram_logic_file, text_logic_file, seq_num, gemini_api_key):
    test_lst = range(2401, 2405)

    with open(diagram_logic_file) as f:
        diagram_logic_forms = json.load(f)
    with open(text_logic_file) as f:
        text_logic_forms = json.load(f)

    combined_logic_forms = {
        pid: diagram_logic_forms[str(pid)]['diagram_logic_forms'] + text_logic_forms[str(pid)]['text_logic_forms']
        for pid in test_lst
    }

    theorem_definitions = {
        1:  "func1_direct_triangle_sum_theorem",
        2:  "func2_indirect_triangle_sum_theorem",
        3:  "func3_isosceles_triangle_theorem_line",
        4:  "func4_isosceles_triangle_theorem_angle",
        5:  "func5_congruent_triangles_theorem_line",
        6:  "func6_congruent_triangles_theorem_angle",
        7:  "func7_radius_equal_theorem",
        8:  "func8_tangent_radius_theorem",
        9:  "func9_center_and_circumference_angle",
        10: "func10_parallel_lines_theorem",
        11: "func11_flat_angle_theorem",
        12: "func12_intersecting_chord_theorem",
        13: "func13_polygon_interior_angles_theorem",
        14: "func14_similar_triangle_theorem",
        15: "func15_angle_bisector_theorem",
        16: "func16_cosine_theorem",
        17: "func17_sine_theorem"
    }

    theorem_prompt = "Theorem definitions:\n" + "\n".join(
        f"{tid}: {desc}" for tid, desc in theorem_definitions.items()
    )

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')

    final = {}

    for pid in tqdm(test_lst):
        logic_text = str(combined_logic_forms[pid])
        input_prompt = (
            theorem_prompt +
            "\n\nProblem Logic Forms:\n" + logic_text +
            "\n\nUsing the theorem definitions above and the provided logic forms, generate candidate theorem sequences that could solve the problem. "
            "Each candidate sequence must be a list of theorem IDs (from 1 to 17). "
            f"Generate exactly {seq_num} candidate sequences. "
            "Output strictly as a Python list of lists. Example: [[16], [10, 14], [10], [14], [10, 16]]"
        )

        max_attempts = 5
        attempt = 0
        responses = []

        while attempt < max_attempts:
            try:
                responses = model.generate_content([input_prompt] * seq_num)
                print(f"\n--- Raw responses for pid {pid} ---")
                for r in responses:
                    print(r.text)
                print("--- End of raw response ---\n")
                break
            except ResourceExhausted:
                attempt += 1
                wait_time = 45
                print(f"Rate limit hit. Waiting {wait_time}s (Attempt {attempt}/{max_attempts})...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Unexpected error for PID {pid}: {e}")
                responses = ['[]'] * seq_num
                break

        output_sequences = []
        for res in responses:
            if isinstance(res, str):
                output_sequences.append(clean_and_parse_response(res))
            else:
                output_sequences.append(clean_and_parse_response(res.text))

        final[str(pid)] = {"id": str(pid), "num_seqs": seq_num, "seq": output_sequences}

    return final

if __name__ == '__main__':
    diagram_logic_file = '../diagram_parser/diagram_logic_forms.json'
    text_logic_file = '../text_parser/text_logic_forms.json'
    output_file = 'results/test/pred_seqs_test_gemini.json'
    SEQ_NUM = 5

    gemini_api_key = "AIzaSyA3ronSnLTaKyLEX7sgG4LSm-EEDYZ1DkI"

    result = evaluate(
        diagram_logic_file=diagram_logic_file,
        text_logic_file=text_logic_file,
        seq_num=SEQ_NUM,
        gemini_api_key=gemini_api_key
    )

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)