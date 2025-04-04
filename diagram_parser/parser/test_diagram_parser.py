import os
import sys
import cv2
import json
from tqdm import tqdm
import argparse
from multiprocess import Pool
import matplotlib.pyplot as plt
import numpy as np
import re

from geosolver.diagram.parse_confident_formulas import parse_confident_formulas
from geosolver.diagram.get_instances import get_all_instances
from geosolver.ontology.ontology_definitions import *

from load_symbol import load_symbol
from use_geosolver import image_to_graph_parse
from parse_symbol import generate_label
from symbol_grounding import parser_to_generate_description
from generate_logic_form import solveSigns, solvePerpendicular, solveParallels, solveLines, solveAngles

def refine_point_labels(graph_parse, point_key_dict, delete_points):
    """
    Refine point label assignment by sorting points based on their spatial positions.
    Points already in point_key_dict are preserved; remaining points are assigned in sorted order.
    """
    points = get_all_instances(graph_parse, 'point')
    valid_points = []
    for pid, pt in points.items():
        if pid not in delete_points:
            valid_points.append((pid, pt.x, pt.y))
    
    # Sort points by x-coordinate, then by y-coordinate
    valid_points.sort(key=lambda x: (x[1], x[2]))
    
    # Start with letters that were already assigned in point_key_dict
    assigned = {}
    available_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for letter, pid in point_key_dict.items():
        assigned[pid] = letter
        if letter in available_letters:
            available_letters.remove(letter)
    
    # Then assign remaining points in sorted order
    for pid, x, y in valid_points:
        if pid not in assigned:
            if available_letters:
                assigned[pid] = available_letters.pop(0)
            else:
                # Fallback: if more than 26 points, assign a custom label.
                assigned[pid] = f"P{pid}"
    return assigned

def improved_multithread_solve(parameters):
    # Unpack parameters
    data_id, box_id, ocr_result, sign_result, data, img, factor = parameters
    log_message = []

    # Parse the diagram with GeoSolver
    graph_parse = image_to_graph_parse(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    points = get_all_instances(graph_parse, 'point')
    
    # Generate logic forms for signs
    delete_points, formula_list = solveSigns(graph_parse, factor, sign_result, log_message)
    known_label = generate_label(graph_parse, ocr_result, delete_points, factor, log_message)
    
    # Generate additional label data and basic logic forms
    point_key_dict = parser_to_generate_description(graph_parse, known_label, delete_points, formula_list, log_message)
    
    # Add logic forms for PointLieOnLine and PointLieOnCircle from confident formulas
    for variable_node in parse_confident_formulas(graph_parse):
        result = variable_node.simple_repr()
        legal = True
        for element in re.split(r'[,\(\)]', result):
            try:
                val = int(element.replace('point_', ''))
                if val in delete_points:
                    legal = False
            except Exception:
                pass
        if legal:
            formula_list.append(variable_node)
    
    # Refine the assignment of letters to points using spatial order
    refined_labels = refine_point_labels(graph_parse, point_key_dict, delete_points)
    
    # Replace point IDs in each logic form with their refined labels
    new_formula_list = []
    logic_forms = []
    for formula in formula_list:
        if formula.is_leaf():
            continue
        tester = lambda x: x.simple_repr() in refined_labels
        gester = lambda x: FormulaNode(Signature(refined_labels[x.simple_repr()], 'point'), [])
        new_formula = formula.replace_signature(tester, gester)
        new_formula_list.append(new_formula)
        logic_forms.append(new_formula.simple_repr())
    
    # Build the answer dictionary
    answer = {}
    answer["id"] = data_id
    answer["log"] = log_message
    answer['point_instances'] = [refined_labels.get(pid, pid) for pid in points.keys()]
    
    # Process line instances: each line is represented as a concatenation of its endpoints
    line_instances = []
    for endpoints in get_all_instances(graph_parse, 'line').keys():
        p1, p2 = endpoints
        label1 = refined_labels.get(f"point_{p1}", f"point_{p1}")
        label2 = refined_labels.get(f"point_{p2}", f"point_{p2}")
        line_instances.append(label1 + label2)
    answer['line_instances'] = line_instances
    
    # Process circle instances similarly (using the center of the circle)
    circle_instances = []
    for center in graph_parse.circle_dict.keys():
        circle_instances.append(refined_labels.get(f"point_{center}", f"point_{center}"))
    answer['circle_instances'] = circle_instances
    
    answer['diagram_logic_forms'] = logic_forms
    answer['point_positions'] = {refined_labels.get(pid, pid): (pt.x, pt.y) for pid, pt in points.items()}
    
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Improved Diagram Parsing using GeoSolver')
    parser.add_argument('--data_path', default='../../data/geometry3k')
    parser.add_argument('--ocr_path', default='../detection_results/ocr_results')
    parser.add_argument('--box_path', default='../detection_results/box_results')
    parser.add_argument('--output_path', default='improved_diagram_logic_form.json')
    
    args = parser.parse_args()
    
    # Process a sample range of test data (e.g., IDs 2401 to 2404)
    test_data = list(range(2401, 2405))
    detection_id = list(map(str, test_data))
    
    ocr_results, sign_results, size_results = load_symbol(detection_id, args.ocr_path, args.box_path)
    
    para_lst = []
    for data_id in test_data:
        box_id = str(data_id)
        input_path = os.path.join(args.data_path, "test", str(data_id))
        json_file = os.path.join(input_path, "data.json")
        with open(json_file, 'r') as f:
            data = json.load(f)
        diagram_path = os.path.join(input_path, "img_diagram.png")
        img = cv2.imread(diagram_path)
        factor = (1, 1)
        if size_results[box_id] is not None:
            factor = (img.shape[1] / size_results[box_id][1], img.shape[0] / size_results[box_id][0])
        para_lst.append((data_id, box_id, ocr_results[box_id], sign_results[box_id], data, img, factor))
    
    solve_list = []
    with tqdm(total=len(para_lst), ncols=80) as t:
        with Pool(10) as p:
            for answer in p.imap_unordered(improved_multithread_solve, para_lst):
                solve_list.append(answer)
                t.update()
    
    solve_list = sorted(solve_list, key=lambda x: int(x['id']))
    final = {}
    for entry in solve_list:
        id = entry["id"]
        del entry["id"]
        final[id] = entry
    with open(args.output_path, "w") as f:
        json.dump(final, f, indent=2)
