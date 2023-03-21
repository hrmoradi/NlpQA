from Libraries import *


def ReturnNotes(typeFile, data_path):
    xlsxPath = data_path
    if typeFile == "smaller":
        xlsxFileName = r"R521_27447_OP_NOTE_102.XLSX"
        temp = pd.read_csv(os.path.join(data_path, "R521_27447_OP_NOTE_102_labels.csv"), dtype={"Label_Start": 'Int64', "Label_end": 'Int64'})

    elif typeFile == "larger":
        xlsxFileName = "TOTAL_KNEE_ARTHROPLASTY__(27447).XLSX"  # One with over 1000
        temp = pd.read_csv(os.path.join(data_path, r"TOTAL_KNEE_ARTHROPLASTY__(27447)_labels.csv"),
                           dtype={"Label_Start": 'Int64', "Label_end": 'Int64'})

    medNotes_dtypes = {"OP_NOTE": str, "AGE at CPT CODE":'Int64', "height in Inches":'Float64', "Weight in KGs":'Float64',
                       "Last recorded BMI":'Float64', "Ethnic_Group":str, "Smoking":str, "Sex":str, "Race":str}

    # Read in medical notes
    xlsx_file_path = os.path.join(xlsxPath, xlsxFileName)
    medicalNotes = pd.read_excel(xlsx_file_path, dtype=medNotes_dtypes, na_values="NULL")

    # Drop rows that are all missing
    medicalNotes = medicalNotes.dropna(axis=0, how="all")
    medicalNotes = medicalNotes.reset_index(drop=True)

    # Remove any whitespaces that have more than one in a row
    medicalNotes["OP_NOTE"] = medicalNotes["OP_NOTE"].apply(lambda x: " ".join(x.split()))

    # Read in labels and combine with medical notes
    temp["CPT Code Date"] = pd.to_datetime(temp["CPT Code Date"])
    medicalNotes = medicalNotes.merge(temp, on=["pat_id", "CPT Code Date"])
    return medicalNotes


def extractModelInfo(Notes):
    model_input = Notes[["pat_id", "Question", "OP_NOTE", "Label", "Raw_Label", "Label_Start"]]

    # For the constraint type question, the current label file has the ground truth CR/PS in Label
    # But the text from OP_NOTE that was used to label it as such is located in Raw_Label
    # So this is rewriting Label with Raw_Label for the Constraint Type question only
    model_input.loc[model_input["Question"] == "What is the constraint type?", "Label"] = model_input.loc[
        model_input["Question"] == "What is the constraint type?", "Raw_Label"]
    model_input = model_input.drop(columns=["Raw_Label"])

    # Rename columns to ones defined in custom feature list for huggingface dataset
    model_input = model_input.rename({"pat_id": "id", "Question": "question",
                                      "OP_NOTE": "context", "Label": "text", "Label_Start": "answer_start"}, axis=1)

    # Remove any observations that do not have label
    model_input = model_input.dropna(subset=["text"], axis=0)
    return model_input

# def stratifyOver(toStratOver):
#     if toStratOver == "questions":
#

def changeOPNote(rawText):
    if re.search(r"(post stabilized)", rawText, re.IGNORECASE):
        compiled = re.compile(r"post stabilized", re.IGNORECASE)
        rawText = compiled.sub("posterior stabilized", rawText)
    #         rawText = re.sub("post stabilized", "posterior stabilized", rawText, flags=re.IGNORECASE)

    if re.search(r"(\bps\b)", rawText, re.IGNORECASE):
        compiled = re.compile(r"\bps\b", re.IGNORECASE)
        rawText = compiled.sub("posterior stabilized", rawText)
    #         rawText = re.sub("\bps\b", "posterior stabilized", rawText, flags=re.IGNORECASE)

    if re.search(r"(\bcr\b)", rawText, re.IGNORECASE):
        compiled = re.compile(r"\bcr\b", re.IGNORECASE)
        rawText = compiled.sub("cruciate retaining", rawText)
    #         rawText = re.sub("\bcr\b", "cruciate retaining", rawText, flags=re.IGNORECASE)

    return rawText

def importModelandTokenizer(modelName, caseVer):
    if modelName.lower() == "distilbert":
        if caseVer == "lowercase":
            if platform.system() == "Windows":
                rawName = "distilbert-base-uncased-distilled-squad"
            elif platform.system() == "Linux":
                rawName = r"/home/dmlee/[models]/distilbert-base-uncased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFDistilBertForQuestionAnswering.from_pretrained(rawName)
        elif caseVer == "uppercase":
            if platform.system() == "Windows":
                rawName = "distilbert-base-cased-distilled-squad"
            elif platform.system() == "Linux":
                rawName = r"/home/dmlee/[models]/distilbert-base-cased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFDistilBertForQuestionAnswering.from_pretrained(rawName)
    elif modelName.lower() == "bert":
        if caseVer == "lowercase":
            if platform.system() == "Windows":
                rawName = "bert-base-uncased"
            elif platform.system() == "Linux":
                rawName = r"/home/dmlee/[models]/bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFBertForQuestionAnswering.from_pretrained(rawName)
        elif caseVer == "uppercase":
            if platform.system() == "Windows":
                rawName = "bert-base-cased"
            elif platform.system() == "Linux":
                rawName = r"/home/dmlee/[models]/bert-base-cased"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFBertForQuestionAnswering.from_pretrained(rawName)

    elif modelName.lower() == "biobert":
        if caseVer == "uppercase" or caseVer == "lowercase":
            if platform.system() == "Windows":
                rawName = "dmis-lab/biobert-v1.1"
            elif platform.system() == "Linux":
                rawName = "dmis-lab/biobert-v1.1"
                # rawName = r"/home/dmlee/[models]/bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFBertForQuestionAnswering.from_pretrained(rawName, from_pt=True)

    elif modelName.lower() == "clinicalbert":
        if caseVer == "uppercase" or caseVer == "lowercase":
            if platform.system() == "Windows":
                rawName = "emilyalsentzer/Bio_ClinicalBERT"
            elif platform.system() == "Linux":
                rawName = "emilyalsentzer/Bio_ClinicalBERT"
                # rawName = r"/home/dmlee/[models]/bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFBertForQuestionAnswering.from_pretrained(rawName, from_pt=True)

    elif modelName.lower() == "gptj":
        if caseVer == "uppercase" or caseVer == "lowercase":
            if platform.system() == "Windows":
                rawName = "EleutherAI/gpt-j-6B"
            elif platform.system() == "Linux":
                rawName = "EleutherAI/gpt-j-6B"
                # rawName = r"/home/dmlee/[models]/bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(rawName)
            model = TFGPTJForQuestionAnswering.from_pretrained(rawName, from_pt=True)

    return tokenizer, model

def preprocess_function(examples, tokenizer, max_length, doc_stride):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples, tokenizer, max_length, doc_stride):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples, list_questions, n_best=20, max_answer_length=30):
    metric = evaluate.load("squad")
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    ## Convert predicted and ground truth labels to lowercase prior to metric calculation
    for i in range(len(predicted_answers)):
        predicted_answers[i]["prediction_text"] = predicted_answers[i]["prediction_text"].lower()
        theoretical_answers[i]["answers"]["text"] = [theoretical_answers[i]["answers"]["text"][0].lower()]

    result_dict = {}
    result_dict["overall"] = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    # Only iterate over the questions present in this batch
    list_questions = [question for question in list_questions if question in examples["question"]]
    for i in range(len(list_questions)):
        list_idx = [j for j in range(len(examples)) if examples["question"][j] == list_questions[i]]
        sub_pred = [predicted_answers[k] for k in list_idx]
        sub_theo = [theoretical_answers[k] for k in list_idx]
        result_dict[list_questions[i]] = metric.compute(predictions=sub_pred, references=sub_theo)

    return result_dict, predicted_answers, theoretical_answers;


def printOverallResults(outputPath, fileName, modelDetails, dataset_dict, trainingDetails, hyperparameters, stats,
                        predicted_answers, execTime, list_questions, test_num_samples):
    """
    :param outputPath:
        path to folder to save all files
    :param fileName:
        name of file to save results (csv with overall results)
    :param n_label:
        For now, the number of CSSRS labels.
    :return:
    """

    n_question = len(set([x["Question"] for x in predicted_answers]))

    if trainingDetails["type"] == "CV":
        outputPath = os.path.join(outputPath, "CV", f"[{numCV} Folds]", f"[{n_question} Questions]")
    elif trainingDetails["type"] == "split":
        outputPath = os.path.join(outputPath, "Split", f"[{n_question} Questions]")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    hours, minutes, seconds = str(execTime).split(":")
    results = pd.DataFrame(
        {"Model": modelDetails["name"], "Case": modelDetails["case"], "Training Dataset": dataset_dict["train"],
         "Split Type": trainingDetails["type"], "Stratification":trainingDetails["strat_on"], "Oversample":trainingDetails["oversample"],
         "Number of Questions": n_question, "Questions Per Model":trainingDetails["model_split"],
         "Overall Exact Match": stats['overall']["exact_match"], "Overall F1 Score": stats['overall']["f1"],
         "Execution Time": f"{hours}H{minutes}M",
         "random.seed": seed, "np seed": seed, "tf seed": seed, "Notes": ""}, index=[0])

    if trainingDetails["type"] == "split":
        results[f"Hyperparameters"] = str(sorted(list(hyperparameters.items()), key=lambda x: x[0][0]))
    else:
        for i in range(numCV):
            results[f"Fold {i + 1} Hyperparameters"] = str(
                sorted(list(hyperparameters[i].items()), key=lambda x: x[0][0]))

    results["Questions"] = str(list_questions)

    list_between = []
    for i in range(len(list_questions)):
        results[f"Q{i + 1} Exact Match"] = stats[list_questions[i]]["exact_match"]
        results[f"Q{i + 1} F1 Score"] = stats[list_questions[i]]["f1"]

        list_between.append(f"Q{i + 1} Exact Match")
        list_between.append(f"Q{i + 1} F1 Score")

    file_path = os.path.join(outputPath, fileName)

    if not os.path.exists(file_path):
        qid = 1
    else:
        temp_df = pd.read_csv(file_path)
        qid = temp_df.iloc[-1, 0] + 1
    results["QID"] = qid

    if trainingDetails["type"] == "split":
        list_before = ["QID", "Model", "Case", "Training Dataset", "Split Type", "Stratification", "Oversample",
                       "Number of Questions", "Questions Per Model", "Questions",
                       "Overall Exact Match", "Overall F1 Score"]

        list_after = ["Hyperparameters", "Execution Time", "random.seed", "np seed", "tf seed", "Notes"]

        results = results[list_before + list_between + list_after]

    file_path = os.path.join(outputPath, fileName)
    results.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))

    if hyperparameters["epochs"] == 1:
        outName = f'[{qid}] Predicted Output - {hyperparameters["epochs"]} Epoch.txt'
    else:
        outName = f'[{qid}]Predicted Output - {hyperparameters["epochs"]} Epochs.txt'

    # Sort predicted answers in alphabetical order in order of Question then ground truth label
    predicted_answers = sorted(predicted_answers, key=lambda x: (x["Question"], x["actual_text"][0], x["id"]))

    with open(os.path.join(outputPath, outName), 'w') as f:
        f.write(f"{'overall', stats['overall'], test_num_samples['overall']}\n")
        for key, value in stats.items():
            if key != "overall":
                f.write(f"{key, value, test_num_samples[key]}\n")
        if trainingDetails["model_split"] == "all":
            f.write(f"One model for all questions\n")
        elif trainingDetails["model_split"] == "one":
            f.write(f"One model for each question\n")
        f.write("\n\n")
        for line in predicted_answers:
            f.write(f"{line}\n")
