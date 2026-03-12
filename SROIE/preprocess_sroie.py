# This script converts the dataset into HuggingFace Dataset format.
import os
import json
import logging
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

LABELS = [
"O",
"B-COMPANY","I-COMPANY",
"B-DATE","I-DATE",
"B-ADDRESS","I-ADDRESS",
"B-TOTAL","I-TOTAL"
]

def normalize_bbox(bbox, width, height):

    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

def polygon_to_box(coords):

    xs = coords[0::2]
    ys = coords[1::2]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    ]

def load_box_file(path):

    tokens = []
    boxes = []
    lines = []
    with open(path, encoding="latin1") as f:
        # loop over lines one by one and skip empty lines
        while line := f.readline():
            if line.strip() == "":
                continue
            lines.append(line)
        # logging.info(f"Loaded {len(lines)} lines from {path}")
        # print(lines)

    for no, line in enumerate(lines):
        parts = line.strip().split(",")
        # print(f"Line {no}: {parts}")
        coords = list(map(int, parts[:8]))
        # print(f"Coords: {coords}")
        text = ",".join(parts[8:]).strip()
        # print(f"Text: '{text}'")

        if text == "":
            continue

        box = polygon_to_box(coords)

        tokens.append(text)
        boxes.append(box)

    return tokens, boxes

def normalize_text(text):
    return text.lower().replace(",", "").replace(".", "").strip()

def split_words(token, bbox):
    """
    Split a line token into word tokens and estimate bounding boxes
    using character-length proportional width.

    token: str
    bbox: [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = bbox

    words = token.split()

    # total width
    total_width = x2 - x1

    # count characters INCLUDING spaces
    total_chars = len(token)

    if total_chars == 0:
        return [token], [bbox]

    # average width of a character
    char_width = total_width / total_chars

    new_tokens = []
    new_boxes = []

    cursor = x1

    for i, word in enumerate(words):

        word_len = len(word)

        word_width = char_width * word_len

        nx1 = int(cursor)
        nx2 = int(cursor + word_width)

        new_tokens.append(word)
        new_boxes.append([nx1, y1, nx2, y2])

        cursor += word_width

        # account for space after word (except last)
        if i < len(words) - 1:
            cursor += char_width

    return new_tokens, new_boxes

def label_tokens(tokens, entities):
    # print(f"Labeling tokens: {tokens} with entities: {entities}")
    labels = ["O"] * len(tokens)

    for i, token in enumerate(tokens):
        for key, entity_value in entities.items():
            stripped_token = token.strip()
            stripped_entity = entity_value.strip()

            if stripped_token == "":
                continue

            # Exclude single CHARACTER tokens from matching
            if len(stripped_token) <= 1:
                continue

            if key.upper() == "COMPANY":
                    # exclude numbers
                try:
                    float(token) # Attempt to convert to a float
                    continue # If successful, it is a number, so continue
                except ValueError:
                    pass # If it raises a ValueError, it is not a number, so we can continue


            if stripped_token in stripped_entity or stripped_entity in stripped_token:
                # print(f"Token {i}: '{token}' matched entity '{key}' with value '{entity_value}'")
                token_array = stripped_token.split()
                # Check if the token starts with the entity value to assign B- or I- label
                if entity_value.startswith(token_array[0]):
                    labels[i] = f"B-{key.upper()}"
                else:
                    labels[i] = f"I-{key.upper()}"
                # print(f"Label assigned: {labels[i]} for token: {tokens[i]}")

    return labels

def label_tokens2(tokens, entities):

    labels = ["O"] * len(tokens)
    print(f"Labeling tokens: {tokens} with entities: {entities}")

    for key, value in entities.items():
        print(f"Processing entity: {key.upper()} with value: {value}")

        value_tokens = value.split()
        print(f"Value tokens: {value_tokens}")

        tokens_str = " ".join(tokens)
        splitted_tokens = tokens_str.split()
        print(f"Splitted tokens: {splitted_tokens}")
        
        for i in range(len(tokens)):

            for x in range(len(splitted_tokens)):
                window = splitted_tokens[x:x+len(value_tokens)]
                print(f"Window: {window}")

                if window == value_tokens:
                    print(f"Match found for entity: {key.upper()} at position {x} of splitted tokens")

                    labels[i] = f"B-{key.upper()}"
                    print(f"Label assigned: {labels[i]} for token: {splitted_tokens[i]}clear")

                    for j in range(1,len(value_tokens)):
                        labels[i+j] = f"I-{key.upper()}"
                        print(f"Label assigned: {labels[i+j]} for token: {splitted_tokens[i+j]}")
                    # if we found a match, we can break out of the loop to avoid multiple matches for the same entity
                    break

    return labels

def company_match(category, token, entity_value):
    # exclude numbers
    try:
        float(token) # Attempt to convert to a float
        return False # If successful, it is a number, so return False
    except ValueError:
        pass # If it raises a ValueError, it is not a number, so we can continue 

    # Check if the normalized token is in the normalized entity value
    if token in entity_value or entity_value in token:
        token_array = token.split()
        if entity_value.startswith(token_array[0]):
            labels[i] = f"B-{key.upper()}"
        else:
            labels[i] = f"I-{key.upper()}"

def process_split(base_path):
    logging.info(f"Processing split at {base_path}")
    box_dir = os.path.join(base_path,"box")
    ent_dir = os.path.join(base_path,"entities")
    img_dir = os.path.join(base_path,"img")

    dataset = []

    for file in tqdm(os.listdir(box_dir)):

        name = file.replace(".txt","")

        box_path = os.path.join(box_dir,file)
        ent_path = os.path.join(ent_dir,file)
        img_path = os.path.join(img_dir,name+".jpg")
        
        tokens,boxes = load_box_file(box_path)

        with open(ent_path) as f:
            entities = json.load(f)

        labels = label_tokens(tokens,entities)

        img = Image.open(img_path)
        width,height = img.size

        norm_boxes = [
            normalize_bbox(b,width,height)
            for b in boxes
        ]

        dataset.append({
            "id":name,
            "tokens":tokens,
            "bboxes":norm_boxes,
            "labels":labels,
            "image_path":img_path
        })

    return dataset

def test():
    tokens, boxes = load_box_file("/home/joel/Desktop/datasets/SROIE2019/test/box/X51006619346.txt")
    with open("/home/joel/Desktop/datasets/SROIE2019/test/entities/X51006619346.txt") as f:
        entities = json.load(f)
    labels = label_tokens(tokens,entities)

    return

def main():
    base_dir = "/home/joel/Desktop/datasets/SROIE2019"

    train = process_split(os.path.join(base_dir, "train"))
    logging.info(f"Saving processed training data to {os.path.join(base_dir, 'sroie_train.json')}")
    with open(os.path.join(base_dir, "sroie_train.json"), "w") as f:
        json.dump(train, f, indent=2)

    test = process_split(os.path.join(base_dir, "test"))
    logging.info(f"Saving processed test data to {os.path.join(base_dir, 'sroie_test.json')}")
    with open(os.path.join(base_dir, "sroie_test.json"), "w") as f:
        json.dump(test, f, indent=2)

if __name__ == "__main__":
    main()