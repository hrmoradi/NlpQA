from Libraries import *
from ModelFunctions import MyTFQuestionAnswering, MyXLnetTFQuestionAnswering, MyLlamaQuestionAnswering


def ReturnNotes(typeFile, data_path, data_details):
    xlsxPath = data_path
    if typeFile == "smaller":
        if data_details["scale_back"] == 2:
            print("Using original file...")
            xlsxFileName = "R521_27447_OP_NOTE_102(Original).XLSX"
        else:
            print("Using corrected file...")
            xlsxFileName = "R521_27447_OP_NOTE_102(Simplified).XLSX"

        temp = pd.read_csv(os.path.join(data_path, "R521_27447_OP_NOTE_102_labels.csv"),
                           dtype={"Label_Start": 'Int64', "Label_Stop": 'Int64', "Raw_Label_Start": 'Int64',
                                  "Raw_Label_Stop": 'Int64'},
                           keep_default_na=False)

    elif typeFile == "larger":
        if data_details["scale_back"] == 2:
            print("Using original file...")
            xlsxFileName = "TOTAL_KNEE_ARTHROPLASTY__(27447)(Original except for 2 lines).XLSX"
        else:
            print("Using corrected file...")
            xlsxFileName = "TOTAL_KNEE_ARTHROPLASTY__(27447)(Simplified).XLSX"

        temp = pd.read_csv(os.path.join(data_path, r"TOTAL_KNEE_ARTHROPLASTY__(27447)_labels.csv"),
                           dtype={"Label_Start": 'Int64', "Label_Stop": 'Int64', "Raw_Label_Start": 'Int64',
                                  "Raw_Label_Stop": 'Int64'},
                           keep_default_na=False)

    medNotes_dtypes = {"OP_NOTE": str, "AGE at CPT CODE": 'Int64', "height in Inches": 'Float64',
                       "Weight in KGs": 'Float64',
                       "Last recorded BMI": 'Float64', "Ethnic_Group": str, "Smoking": str, "Sex": str, "Race": str}

    # If 3 questions, remove patella labels
    if str(data_details["num_questions"]) == "3":
        temp = temp.loc[temp["Question"] != "Is there patella resurfacing?", :]

    # Read in medical notes
    xlsx_file_path = os.path.join(xlsxPath, xlsxFileName)
    medicalNotes = pd.read_excel(xlsx_file_path, dtype=medNotes_dtypes, na_values="NULL")
    medicalNotes = medicalNotes[~medicalNotes["pat_id"].isna()]

    # Drop rows that are all missing
    medicalNotes = medicalNotes.dropna(axis=0, how="all")
    medicalNotes = medicalNotes.reset_index(drop=True)

    # Remove any whitespaces that have more than one in a row
    medicalNotes["OP_NOTE"] = medicalNotes["OP_NOTE"].apply(lambda x: " ".join(x.split()))

    # Read in labels and combine with medical notes
    temp["CPT Code Date"] = pd.to_datetime(temp["CPT Code Date"])
    medicalNotes = medicalNotes.merge(temp, on=["pat_id", "CPT Code Date"])
    return medicalNotes


def extractModelInfo(Notes, data_details, onlyUntrained=False):
    model_input = Notes[
        ["pat_id", "Question", "OP_NOTE", "Label", "Raw_Label", "Label_Start", "Label_Stop", "Raw_Label_Start",
         "Raw_Label_Stop"]]

    # For constraint type, only need to switch 'Label' and 'Raw_Label'
    model_input.loc[model_input["Question"] == "What is the constraint type?", "Label"] = model_input.loc[
        model_input["Question"] == "What is the constraint type?", "Raw_Label"]

    if str(data_details["num_questions"]) == "4":
        # For patellar resurfacing, need to switch 'Label', 'Label_Start', 'Label_Stop' w/
        # 'Raw_Label', 'Raw_Label_Start', 'Raw_Label_Stop'
        pat = model_input.loc[model_input["Question"] == "Is there patella resurfacing?", :].copy(deep=True)
        model_input = model_input.loc[model_input["Question"] != "Is there patella resurfacing?", :]

        pat.loc[pat["Question"] == "Is there patella resurfacing?", 'Label'] = pat.loc[
            pat["Question"] == "Is there patella resurfacing?", 'Raw_Label']
        pat.loc[pat["Question"] == "Is there patella resurfacing?", 'Label_Start'] = pat.loc[
            pat["Question"] == "Is there patella resurfacing?", 'Raw_Label_Start']
        pat.loc[pat["Question"] == "Is there patella resurfacing?", 'Label_Stop'] = pat.loc[
            pat["Question"] == "Is there patella resurfacing?", 'Raw_Label_Stop']

        model_input = model_input.drop(columns=['Label_Stop', 'Raw_Label', 'Raw_Label_Start', 'Raw_Label_Stop'])
        pat = pat.drop(columns=['Label_Stop', 'Raw_Label', 'Raw_Label_Start', 'Raw_Label_Stop'])
        # Rename columns to ones defined in custom feature list for huggingface dataset
        model_input = model_input.rename({"pat_id": "id", "Question": "question",
                                          "OP_NOTE": "context", "Label": "text", "Label_Start": "answer_start"}, axis=1)
        pat = pat.rename({"pat_id": "id", "Question": "question",
                          "OP_NOTE": "context", "Label": "text", "Label_Start": "answer_start"}, axis=1)

    elif str(data_details["num_questions"]) == "3":
        model_input = model_input.drop(columns=['Label_Stop', 'Raw_Label', 'Raw_Label_Start', 'Raw_Label_Stop'])
        model_input = model_input.rename({"pat_id": "id", "Question": "question",
                                          "OP_NOTE": "context", "Label": "text", "Label_Start": "answer_start"}, axis=1)

    # Remove any observations that do not have label
    if not onlyUntrained:
        model_input = model_input.loc[model_input["text"] != ""]

    if str(data_details["num_questions"]) == "4":
        model_input = pd.concat([model_input, pat])
        model_input = model_input.convert_dtypes()

    return model_input


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

def getCaseVersion(modelName):
    uncasedVersions = ["distilbert-base-uncased", "distilbert-base-uncased-distilled-squad",
                       "bert-base-uncased"]
    casedVersions = ["distilbert-base-cased", "distilbert-base-cased-distilled-squad",
                     "bert-base-cased", "dmis-lab/biobert-v1.1", "emilyalsentzer/Bio_ClinicalBERT",
                     "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                     "roberta-base", "roberta-large", r"decapoda-research/llama-7b-hf", "xlnet-base-cased", "xlnet-large-cased",
                     "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    if modelName in uncasedVersions:
        caseVer = "lowercase"
    elif modelName in casedVersions:
        caseVer = "uppercase"

    if caseVer is None:
        raise TypeError

    return caseVer

def importModelandTokenizer(modelName):
    tfAutoModelsQA = ["distilbert-base-uncased", "distilbert-base-cased",
                      "distilbert-base-uncased-distilled-squad","distilbert-base-cased-distilled-squad",
                      "bert-base-uncased", "bert-base-cased", "dmis-lab/biobert-v1.1", "emilyalsentzer/Bio_ClinicalBERT",
                      "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext","roberta-base", "roberta-large",
                      "xlnet-base-cased", "xlnet-large-cased", "EleutherAI/gpt-j-6B"]

    tokenizer = AutoTokenizer.from_pretrained(modelName)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if modelName in tfAutoModelsQA:
        model = TFAutoModelForQuestionAnswering.from_pretrained(modelName)

    return tokenizer, model


def importTokenizer(modelName):
    if "llama" in modelName.lower():
        if platform.system() == "Windows":
            model_path = r"D:\zProjects\[Models]\llama-7b-hf"
        elif platform.system() == "Linux":
            model_path = r"/home/dmlee892/models/llama-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    elif "gpt2" in modelName.lower():
        if "gpt2-xl" in modelName.lower():
            if platform.system() == "Windows":
                model_path = r"D:\zProjects\[Models]\gpt2-xl"
            elif platform.system() == "Linux":
                pass
        else:
            model_path = modelName

        bos = '<|startoftext|>'
        eos = '<|endoftext|>'
        pad = '<|pad|>'
        special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}

        tokenizer_orig = AutoTokenizer.from_pretrained(model_path)  # transformer library
        tokenizer_orig.add_special_tokens(
            special_tokens_dict)  # with this, you don't have to manually define the new tokens' ids
        tokenizer = Tokenizer.from_pretrained(model_path)  # tokenizer library
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", tokenizer_orig.bos_token_id), ("[SEP]", tokenizer_orig.eos_token_id)],
        )
        tokenizer = GPT2TokenizerFast(
            tokenizer_object=tokenizer)  # transformer library again but now with post processing
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    else:
        tokenizer = AutoTokenizer.from_pretrained(modelName)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def importCustomModel(modelName):
    if "llama" in modelName.lower():
        if platform.system() == "Windows":
            model_path = r"D:\zProjects\[Models]\llama-7b-hf"
        elif platform.system() == "Linux":
            # model_path = r"/data8/[HF Models]/llama-7b-hf"
            model_path = modelName
        model = MyLlamaQuestionAnswering(modelName, model_path)
    elif "gpt2" in modelName.lower():

        if "gpt2-xl" in modelName.lower():
            if platform.system() == "Windows":
                model_path = r"D:\zProjects\[Models]\gpt2-xl"
            elif platform.system() == "Linux":
                #model_path = r"/data8/[HF Models]/gpt2-xl"
                model_path = modelName
        else:
            model_path = modelName
        model = MyTFQuestionAnswering(modelName, model_path)
    elif "xlnet" in modelName.lower():
        model_path = modelName
        model = MyXLnetTFQuestionAnswering(modelName, model_path)
    else:
        model_path = modelName
        model = MyTFQuestionAnswering(modelName, model_path)
    return model

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

def compute_metricsPerBatch(metric, start_logits, end_logits, features, examples, list_questions, n_best=20, max_answer_length=30):
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

    metric.add_batch(predictions=predicted_answers, references=theoretical_answers)

    return metric, predicted_answers, theoretical_answers;

# This function calculates metrics for Patella question differently
# For all negative questions, it auto assigns correctly predicted
def compute_metrics2(start_logits, end_logits, features, examples, list_questions, n_best=20, max_answer_length=30):
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

    # get indices of negative examples and replace predicted text
    neg_idx = [j for j in range(len(theoretical_answers)) if theoretical_answers[j]['answers']['text'][0] == ""]
    for k in range(len(predicted_answers)):
        # Must replace empty string with some string otherwise F1 score is not calculated correctly
        if k in neg_idx:
            predicted_answers[k]["prediction_text"] = "N/A"
            theoretical_answers[k]["answers"]["text"] = ["N/A"]

    result_dict = {}
    result_dict["overall"] = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    # is there patella resurfacing?
    list_of_pos_names = ["oval 3 peg patella", "patella button"]

    # Only iterate over the questions present in this batch
    list_questions = [question for question in list_questions if question in examples["question"]]
    for i in range(len(list_questions)):

        if list_questions[i] != "is there patella resurfacing?":
            list_idx = [j for j in range(len(examples)) if examples["question"][j] == list_questions[i]]
            sub_pred = [predicted_answers[k] for k in list_idx]
            sub_theo = [theoretical_answers[k] for k in list_idx]
            result_dict[list_questions[i]] = metric.compute(predictions=sub_pred, references=sub_theo)
        # For patella question
        else:
            list_idx = [j for j in range(len(examples)) if examples["question"][j] == list_questions[i]]
            sub_pred = [predicted_answers[k] for k in list_idx]
            sub_theo = [theoretical_answers[k] for k in list_idx]

            # get indices of negative examples and replace predicted text
            neg_idx = [j for j in range(len(sub_theo)) if sub_theo[j]['answers']['text'][0] == ""]
            for k in range(len(sub_pred)):
                # Must replace empty string with some string otherwise F1 score is not calculated correctly
                if k in neg_idx:
                    sub_pred[k]["prediction_text"] = "N/A"
                    sub_theo[k]["answers"]["text"] = ["N/A"]
            #             print(sub_pred)
            #             print("----------")
            #             print(sub_theo)
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
         "Train Kept":trainingDetails["train_num"],
         "Number of Questions": n_question, "Questions Per Model":trainingDetails["model_split"],
         "Overall Exact Match": stats['overall']["exact_match"], "Overall F1 Score": stats['overall']["f1"],
         "Execution Time": f"{hours}H{minutes}M",
         "random.seed": seed, "np seed": seed, "tf seed": seed, "Notes": trainingDetails["notes"]}, index=[0])

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
                       "Train Kept", "Number of Questions", "Questions Per Model", "Questions",
                       "Overall Exact Match", "Overall F1 Score"]

        list_after = ["Hyperparameters", "Execution Time", "random.seed", "np seed", "tf seed", "Notes"]

        results = results[list_before + list_between + list_after]

    file_path = os.path.join(outputPath, fileName)
    results.to_csv(file_path, mode="a", index=False, header=not os.path.exists(file_path))

    if hyperparameters["epochs"] == 1:
        outName = f'[{qid}] Predicted Output - {hyperparameters["epochs"]} Epoch.txt'
    else:
        outName = f'[{qid}] Predicted Output - {hyperparameters["epochs"]} Epochs.txt'

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


def getTrainIndices(train_dict, wanted_percent=1):
    ans_idx_dict = {}
    ans_cnt_dict = {}
    for idx, ans in enumerate(train_dict['answers']):
        curr_ans = ans['text'][0]
        if curr_ans not in ans_idx_dict.keys():
            ans_idx_dict[curr_ans] = [idx]
            ans_cnt_dict[curr_ans] = 1
        else:
            ans_idx_dict[curr_ans].append(idx)
            ans_cnt_dict[curr_ans] += 1

    ans_cnt_dict = dict(sorted(ans_cnt_dict.items(), key=lambda item: item[1], reverse=True))
    print(f"Original Numbers: {ans_cnt_dict}")

    wanted_amount = ceil(wanted_percent * len(train_dict))
    print(f"Original Amount: {len(train_dict)}")
    print(f"Wanted Amount: {wanted_amount}")

    wnt_cnt_dict = {k: ceil((wanted_amount * v) / len(train_dict)) for k, v in ans_cnt_dict.items()}
    diff = sum(list(wnt_cnt_dict.values())) - wanted_amount
    if diff > 0:
        # Removes the excess number from the answer with the largest number of samples
        wnt_cnt_dict[next(iter(wnt_cnt_dict))] -= diff
    print(f"Downsampled Numbers: {wnt_cnt_dict}")
    select_idx = []
    for k, v in wnt_cnt_dict.items():
        random.seed(seed)
        select_idx += random.sample(ans_idx_dict[k], v)

    return select_idx


def split_by_pt(to_split, split_percent=0.6):
    num_orig = len(to_split)
    uniq_ids = []
    [uniq_ids.append(x.rsplit("_", 1)[0]) for x in to_split["id"] if x.rsplit("_", 1)[0] not in uniq_ids]
    num_uniq_ids = len(uniq_ids)

    set1_want_num = ceil(num_uniq_ids * split_percent)
    set2_want_num = num_uniq_ids - set1_want_num

    random.seed(seed)
    set1_sel_ids = random.sample(uniq_ids, set1_want_num)
    set2_sel_ids = []
    [set2_sel_ids.append(x) for x in uniq_ids if x not in set1_sel_ids]

    set1 = to_split.filter(lambda ex: ex["id"].rsplit("_", 1)[0] in set1_sel_ids)
    set2 = to_split.filter(lambda ex: ex["id"].rsplit("_", 1)[0] in set2_sel_ids)

    return set1, set2


# Copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat