import collections


def prepare_new_column(raw_dataset, prefix_token = "Rot categories:"):
    values = []

    for data in raw_dataset:
        rot_id = "-".join(data["rot-id"].split("/")[-2:])
        values.append(f"{data['situation']}. {prefix_token} {rot_id}, {data['rot-categorization']}, {data['rot-moral-foundations']}, {data['rot-char-targeting']}")
    
    return values

def get_situations_with_not_unique_rot_categories(raw_dataset):
    situationsToRotId = collections.defaultdict(list)
    for data in raw_dataset:
        rot_id = "-".join(data["rot-id"].split("/")[-2:])
        situationsToRotId[data["situation-short-id"].split("/")[-1]].append((rot_id, data["rot-categorization"], 
                                                                            data["rot-moral-foundations"], data["rot-judgment"], 
                                                                            data['rot-char-targeting'], data["rot-agree"]))
    situations = list(situationsToRotId.keys())

    situationsToRotCategories = collections.defaultdict(set)
    for situation, rotTags in situationsToRotId.items():
        for rotTag in rotTags:
            situationsToRotCategories[situation].add((rotTag[0], rotTag[1], rotTag[2], rotTag[-2]))
    count = 0
    notUniqueRotCategoriesSituations = set()

    for situation in situations:
        if len(situationsToRotCategories[situation]) != len(situationsToRotId[situation]):
            notUniqueRotCategoriesSituations.add(situation)
            count += 1
    print(f"Number of situations with not unique ROT categories: {count}")
    return notUniqueRotCategoriesSituations

def preprocess_function_seq_to_seq(examples, max_input_length = 256, max_target_length = 64):
    model_inputs = tokenizer(
        examples["situation"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["rot"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
