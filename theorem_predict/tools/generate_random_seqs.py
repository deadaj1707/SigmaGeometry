import os
import json
import random
import numpy as np

from numpy.random import seed
from multiprocessing import Pool
from tqdm import tqdm

seed(1234)
random.seed(1234)


class TheoremSets:
    def __init__(self):
        self.id2name = {
            1:  "func1_direct_triangle_sum_theorem",
            2:  "func2_indirect_triangle_sum_theorem",  # The sum of three angles in a triangle is 180
            3:  "func3_isosceles_triangle_theorem_line",
            4:  "func4_isosceles_triangle_theorem_angle", # The base angles of an isosceles triangle are equal.
            5:  "func5_congruent_triangles_theorem_line",
            6:  "func6_congruent_triangles_theorem_angle",
            7:  "func7_radius_equal_theorem",
            8:  "func8_tangent_radius_theorem",
            9:  "func9_center_and_circumference_angle",
            10: "func10_parallel_lines_theorem",
            11: "func11_flat_angle_theorem",  # If point O lies on segment (A, B), then AOC + COB = 180.
            12: "func12_intersecting_chord_theorem",
            13: "func13_polygon_interior_angles_theorem",  # Equations in Quadrilateral
            14: "func14_similar_triangle_theorem",
            15: "func15_angle_bisector_theorem",
            16: "func16_cosine_theorem",
            17: "func17_sine_theorem"
        }

        self.basic_rules = [1, 2, 3, 4, 5, 6, 10, 11, 13, 14, 15, 16, 17]

        self.feat2ids = {
            "circle": [7, 8, 9, 12],
            "arc": [7, 8, 9, 12],
            "sector": [7, 8, 9, 12],
            "triangle": [1, 2, 3, 4, 5, 6, 14, 15, 16, 17],
            # Other features commented out
            # "polygon": [13],
            # "quadrilateral": [13],
            # "parallelogram": [13],
            # "square": [13],
            # "rectangle": [13],
            # "rhombus": [13],
            # "trapezoid": [13],
            # "kite": [13],
        }

    def find_required(self, text, graph_type, goal_type):
        """
        If a text matches a theorem with all the features, the theorem is required.
        """
        required_rules = []

        # func3_isosceles_triangle_theorem_line
        if "isosceles" in text and ('triangle' in text or 'triangle' in graph_type):
            required_rules.append(3)

        # func4_isosceles_triangle_theorem_angle
        if "isosceles" in text and ('triangle' in text or 'triangle' in graph_type):
            required_rules.append(4)

        # func5_congruent_triangles_theorem_line
        if "congruent" in text and ('triangle' in text or 'triangle' in graph_type):
            required_rules.append(5)

        # func6_congruent_triangles_theorem_angle
        if "congruent" in text and ('triangle' in text or 'triangle' in graph_type):
            required_rules.append(6)

        # func8_tangent_radius_theorem
        if "tangent" in text and ('circle' in text or 'circle' in graph_type):
            required_rules.append(8)

        # func10_parallel_lines_theorem
        if "parallel" in text:
            required_rules.append(10)

        # func12_intersecting_chord_theorem
        if "intersect" in text and ('circle' in text or 'circle' in graph_type):
            required_rules.append(12)

        # func14_similar_triangle_theorem
        if "similar" in text and ('triangle' in text or 'triangle' in graph_type):
            required_rules.append(14)

        # func15_angle_bisector_theorem
        if "bisector" in text:
            required_rules.append(15)

        # func16_cosine_theorem
        if "cos(" in text:
            required_rules.append(16)

        # func17_sine_theorem
        if "sin(" in text:
            required_rules.append(17)

        return list(set(required_rules))  # remove redundant candidates

    def find_possible(self, text, graph_type, goal_type):
        possible_rules = self.basic_rules.copy()

        for key, theos in self.feat2ids.items():
            if key in text.lower():
                possible_rules += theos

        for graph in graph_type:
            if graph in self.feat2ids.keys():
                possible_rules += self.feat2ids[graph]

        return list(set(possible_rules))  # remove redundant candidates


def find_one_sequence(text, max_len, graph_type, goal_type):
    """Find one random sequence for the text."""
    # find theorems candidates using the global theorem_sets instance
    required_rules = theorem_sets.find_required(text, graph_type, goal_type)
    possible_rules = theorem_sets.find_possible(text, graph_type, goal_type)

    # find a random sequence length (fix deprecation warning by indexing the array)
    mu, sigma = max_len // 2, max_len // 6
    seq_len = int(np.random.randn(1)[0] * sigma + mu)
    seq_len = max(seq_len, len(required_rules), 3)  # assure seq_len >= 3

    # generate a random sequence
    candidate_rules = list(set(required_rules + possible_rules))
    random_seq = required_rules + [random.choice(candidate_rules) for _ in range(seq_len - len(required_rules))]
    random.shuffle(random_seq)

    return random_seq


def generate_sequences(text, max_len=30, num=10, graph_type=None, goal_type=None):
    """Find N random sequences for the text."""
    text = text.lower()
    graph_type = [graph.lower() for graph in graph_type]
    goal_type = [goal.lower() for goal in goal_type]
    sequences = [find_one_sequence(text, max_len, graph_type, goal_type) for _ in range(num)]
    return sequences


def predict(pid):
    # Determine data split based on pid
    if pid in range(2101):
        split = 'train'
    elif pid in range(2101, 2401):
        split = 'val'
    else:
        split = 'test'

    pid = str(pid)

    # Read logic form data from globals loaded in the initializer
    text_logics = text_logic_tables[pid]["text_logic_forms"]
    diagram_logics = diagram_logic_tables[pid]["diagram_logic_forms"]
    annot_id = text_logic_tables[pid]["annot_id"]
    assert annot_id == diagram_logic_tables[pid]["annot_id"]

    # Read problem data
    ANSWER_INPUT_PATH = os.path.join(DATA_INPUT_PATH, split, pid, "data.json")
    with open(ANSWER_INPUT_PATH, "r") as f:
        data = json.load(f)
    graph_type = data['problem_type_graph']  # e.g., ["Line"]
    goal_type = data['problem_type_goal']     # e.g., ["Angle"]
    assert pid == str(data['id'])

    problem_text = " ".join(diagram_logics + text_logics)
    sequences = generate_sequences(problem_text, MAX_LEN, SEQ_NUM, graph_type, goal_type)

    # Build output structure
    out_data = {}
    out_data["id"] = pid
    out_data["num_seqs"] = len(sequences)

    if split in ['train', 'val']:
        out_data["seqs"] = sequences
    else:
        # For test set, ensure exactly one sequence is returned
        assert len(sequences) == 1
        if SEQ_TYPE == 'random':
            seq = random.sample(list(range(1, 18)) * 10, 100)
            out_data["seq"] = seq
        elif SEQ_TYPE == 'fixed':
            seq = list(range(1, 18)) * 10
            seq = seq[:100]
            out_data["seq"] = seq
        else:
            out_data["seq"] = sequences[0]

    return out_data


def init_worker_all(data_input_path, text_input_path, diagram_input_path, max_len, seq_num, seq_type):
    """
    Initializer for each worker process. Sets up global variables so that the workers can access
    the data and parameters.
    """
    global DATA_INPUT_PATH, MAX_LEN, SEQ_NUM, SEQ_TYPE, text_logic_tables, diagram_logic_tables, theorem_sets
    DATA_INPUT_PATH = data_input_path
    MAX_LEN = max_len
    SEQ_NUM = seq_num
    SEQ_TYPE = seq_type

    with open(text_input_path, "r") as f1:
        text_logic_tables = json.load(f1)
    with open(diagram_input_path, "r") as f2:
        diagram_logic_tables = json.load(f2)
    theorem_sets = TheoremSets()


if __name__ == '__main__':
    # Set paths
    DATA_INPUT_PATH = "../../data/geometry3k"
    TEXT_INPUT_PATH = "../../data/geometry3k/logic_forms/text_logic_forms_annot_dissolved.json"  # human annotated
    DIAGRAM_INPUT_PATH = "../../data/geometry3k/logic_forms/diagram_logic_forms_annot.json"      # human annotated

    ## ======================== One Example ======================== ##
    print('One example:')
    text = "Equal(Angle(A,B,C),90), Find(AreaOf(Triangle(A,B,C)))."
    # For this one example, we don't need multiprocessing so we can initialize theorem_sets directly.
    theorem_sets = TheoremSets()
    sequences = generate_sequences(text, 15, 10, ['triangle'], ['area'])
    print('question text:', text)
    print('predicted theorem sequences:')
    for seq in sequences:
        print(len(seq), seq)

    ## ======================== Train Set ======================== ##
    print("\nGenerating random theorem sequences for training set...")
    MODE = "Train"
    SEQ_NUM = 100
    MAX_LEN = 30
    TRAIN_PID_RANGE = range(2401)  # Adjust as needed

    # Use an initializer to load JSON data and set globals in each worker process
    with Pool(20, initializer=init_worker_all,
              initargs=(DATA_INPUT_PATH, TEXT_INPUT_PATH, DIAGRAM_INPUT_PATH, MAX_LEN, SEQ_NUM, "")) as p:
        pred_list = p.map(predict, TRAIN_PID_RANGE)

    results = {}
    for data in pred_list:
        pid = data['id']
        # For train/val, sort sequences by their length
        data['seqs'] = sorted(data['seqs'], key=lambda x: len(x))
        results[pid] = data

    JSON_FILE = "../results/train/pred_seqs_train_l{}_n{}_template.json".format(MAX_LEN, SEQ_NUM)
    print("Saving generated theorem sequences to {}".format(JSON_FILE))
    with open(JSON_FILE, 'w') as f:
        json.dump(results, f, indent=2, separators=(',', ': '))

    ## ======================== Test Set ======================== ##
    MODE = "Test"
    SEQ_NUM = 1
    MAX_LEN = 100

    for SEQ_TYPE in ['random', 'fixed']:
        results = {}
        TEST_PID_RANGE = range(2401, 2405)  # Adjust range as needed

        # For test set, pass SEQ_TYPE to the initializer so workers know which test mode to use.
        with Pool(20, initializer=init_worker_all,
                  initargs=(DATA_INPUT_PATH, TEXT_INPUT_PATH, DIAGRAM_INPUT_PATH, MAX_LEN, SEQ_NUM, SEQ_TYPE)) as p:
            pred_list = p.map(predict, TEST_PID_RANGE)

        for data in pred_list:
            pid = data['id']
            results[pid] = data

        JSON_FILE = "../results/test/pred_seqs_test_l{}_{}.json".format(MAX_LEN, SEQ_TYPE)
        print("Saving generated theorem sequences to {}".format(JSON_FILE))
        with open(JSON_FILE, 'w') as f:
            json.dump(results, f, indent=2, separators=(',', ': '))
