from Libraries import *
from transformers import TFAutoModel
from HelperFunctions import ReturnNotes, extractModelInfo, changeOPNote, importModelandTokenizer, importTokenizer, \
    preprocess_function, preprocess_validation_examples, compute_metrics, printOverallResults, importCustomModel, getCaseVersion, \
    compute_metricsPerBatch, compute_metrics2
from DataProcessing import constructData
from ModelFunctions import MyTFQuestionAnswering


def runModel(outputPath, data_ds, ds_dict, list_ques, modelInfo, trainingDetails, hyperparameters):
    # Import tokenizer and model
    if trainingDetails["modelType"] != "custom":
        tokenizer, model = importModelandTokenizer(modelInfo["name"])
    else:
        tokenizer = importTokenizer(modelInfo["name"])
        model = importCustomModel(modelInfo["name"])
        if "gpt2" in modelInfo["name"].lower():
            model.model.resize_token_embeddings(len(tokenizer))


    # The maximum length of a feature (question and context)
    max_length = hyperparameters["max_length"]
    # The authorized overlap between two part of the context when splitting
    doc_stride = hyperparameters["doc_stride"]

    # Tokenize inputs for training
    train_set = data_ds["train"].map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=data_ds["train"].column_names)

    # Tokenize inputs for validation
    val_set = data_ds["val"].map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=data_ds["val"].column_names)

    # Tokenize inputs for test/evaluation set
    validation_dataset = data_ds["test"].map(
        preprocess_validation_examples,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=data_ds['test'].column_names,
    )
    test_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])


    # Convert to format useable with tensorflow
    train_set = train_set.to_tf_dataset(
        columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
        #     label_cols=["start_positions", "end_positions"],
        batch_size=hyperparameters["batch_size"],
        shuffle=False)

    val_set = val_set.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["start_positions", "end_positions"],
        batch_size=hyperparameters["batch_size"],
        shuffle=False)


    ## Use below if using GPU, otherwise leave commented out
    # keras.mixed_precision.set_global_policy("mixed_float16")

    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])
    model.compile(optimizer=optimizer)
    model.fit(train_set, epochs=hyperparameters["epochs"])

    if "xlnet" not in modelInfo["name"]:
        test_set = test_set.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            batch_size=hyperparameters["batch_size"],
            shuffle=False)

        # Get starting and ending logits
        outputs = model.predict(test_set)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Evaluate test
        if len(list_ques) == 4:
            eval_metrics, pred_ans, act_ans = compute_metrics2(start_logits, end_logits, validation_dataset, data_ds["test"],
                                                              list_ques)
        else:
            eval_metrics, pred_ans, act_ans = compute_metrics(start_logits, end_logits, validation_dataset, data_ds["test"],
                                                              list_ques)
    else:
        list_lens = pd.Series(validation_dataset["example_id"]).value_counts(sort=False).tolist()
        dict_question_abbr = {question: "".join([word[0].upper() for word in question.split()]) for question in
                              list_ques}

        # Create batches of whole examples
        cut_points = [hyperparameters["batch_size"] * x for x in
                      list(range((len(list_lens) // hyperparameters["batch_size"]) + 1))]
        cut_points.append(len(list_lens))

        prev_point = 0
        new_list_lens = []
        for i in range(1, len(cut_points)):
            temp = list_lens[prev_point:cut_points[i]]
            if sum(temp) > 0:
                new_list_lens.append(sum(temp))
            prev_point = cut_points[i]
        list_lens = new_list_lens

        eval_metrics = {}

        metric = evaluate.load("squad")
        cum_idx = 0
        pred_ans = []
        act_ans = []
        # for (batch, values) in test_set[cum_idx:list_lens[curr_idx]]:
        print(len(list_lens))
        for i in range(len(list_lens)):
            batch = test_set.select(range(cum_idx, cum_idx + list_lens[i])).to_tf_dataset(
                batch_size=hyperparameters["batch_size"])
            ex_id = list(set(validation_dataset.select(range(cum_idx, cum_idx + list_lens[i]))['example_id']))
            if len(ex_id) > hyperparameters["batch_size"] and i != len(list_lens)-1:
                print(ex_id)
                print(f"Length of batch: {len(ex_id)}")
                print(f"Intended batch_size: {hyperparameters['batch_size']}")
                print(f"More than intended example ids in list. There should only be {hyperparameters['batch_size']}. Exiting...")
                exit(1)

            # print(next(iter(batch)))
            outputs = model.predict(batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            metric, temp_pred_ans, temp_act_ans = compute_metricsPerBatch(metric, start_logits, end_logits,
                                                         validation_dataset.filter(
                                                             lambda row: row['example_id'] in ex_id),
                                                         data_ds["test"].filter(lambda row: row['id'] in ex_id),
                                                         list_ques)
            pred_ans += temp_pred_ans
            act_ans += temp_act_ans

            cum_idx += list_lens[i]

        eval_metrics["overall"] = metric.compute()
        for key, values in dict_question_abbr.items():
            temp_metric = evaluate.load("squad")
            eval_metrics[key] = temp_metric.compute(predictions=[row for row in pred_ans if values in row['id']],
                                               references=[row for row in act_ans if values in row['id']])

    return [eval_metrics, pred_ans, act_ans]

def Pipeline(outputPath, ds_dict, modelInfo, dataDetails, trainingDetails, hyperparameters):
    startTime = datetime.now()

    # get whether model is cased or uncased
    modelInfo["case"] = getCaseVersion(modelInfo["name"])

    if dataDetails["Use_prebuilt_labels"] == "yes":

        # Get train set
        medicalNotes_train = ReturnNotes(ds_dict["train"], dataPath, dataDetails)
        # medicalNotes_train = medicalNotes_train.dropna(axis=0, subset=["Label_Stop"])
        # medicalNotes_train = medicalNotes_train.reset_index(drop=True)

    elif dataDetails["Use_prebuilt_labels"] == "no":
        medicalNotes_train = constructData(dataPath, ds_dict["train"], dataDetails)

    # Modify pat_id to be concatenation of pat_id and CPT CODE
    # since patient can have more than OP Note associated with them
    medicalNotes_train["pat_id"] = medicalNotes_train.apply(
        lambda x: x["pat_id"] + "_" + x["CPT Code Date"].strftime("%Y%m%d"), axis=1)

    if dataDetails["scale_back"] == 0:
        # Change appropriate strings such as 'cr' to 'cruciate retaining' to match labels
        medicalNotes_train["OP_NOTE"] = medicalNotes_train["OP_NOTE"].apply(lambda x: changeOPNote(x))

    model_input_train = extractModelInfo(medicalNotes_train, dataDetails)
    model_input_train = model_input_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    if modelInfo["case"] == "lowercase":
        model_input_train[["question", "context", "text"]] = model_input_train[["question", "context", "text"]].astype(str).apply(lambda col: col.str.lower())

    # Convert to correct nesting
    model_input_train["answers"] = model_input_train.apply(lambda x: {"text": [x["text"]], "answer_start": [x["answer_start"]]},
                                                           axis=1)
    # Drop unncessary columns
    model_input_train = model_input_train.drop(["text", "answer_start"], axis=1)
    model_input_train = model_input_train.reset_index(drop=True)

    # Assign each question to individual id by adding capitalized first letter of each word in question
    model_input_train["id"] = model_input_train.apply(
        lambda x: x["id"] + "_" + "".join([word[0].upper() for word in x["question"].split()]), axis=1)

    #########################################################################################

    # Get List of Questions
    list_ques = sorted(model_input_train["question"].unique().tolist())

    num_ques = len(list_ques)
    question_dict = {list_ques[ques_idx]: ques_idx for ques_idx in range(len(list_ques))}
    inverse_question_dict = {value: key for key, value in question_dict.items()}

    if trainingDetails["strat_on"] == "question" or trainingDetails["strat_on"] == "questions":
        num_strat_classes = num_ques
        model_input_train["strat_id"] = model_input_train["question"].apply(lambda x: question_dict[x])
    elif trainingDetails["strat_on"] == "answer" or trainingDetails["strat_on"] == "answers":

        list_ans = model_input_train["answers"].apply(lambda x: x["text"][0]).unique().tolist()

        num_ans = len(list_ans)
        num_strat_classes = num_ans

        answer_dict = {list_ans[ans_idx]: ans_idx for ans_idx in range(len(list_ans))}
        inverse_answer_dict = {value: key for key, value in answer_dict.items()}
        model_input_train["strat_id"] = model_input_train["answers"].apply(lambda x: answer_dict[x["text"][0]])


    # Create 'check' column to split off the negative patellar rows to append to test set
    model_input_train["check"] = model_input_train["answers"].apply(lambda x: True if x["text"][0] == "" else False)

    pat_neg = model_input_train.loc[(model_input_train["check"] == True), :].copy(deep=True)
    model_input_train = model_input_train.loc[(model_input_train["check"] != True), :]

    pat_neg = pat_neg.drop(columns=["check"])
    model_input_train = model_input_train.drop(columns=["check"])


    if trainingDetails["strat_on"] != "none":
        # Feature List for dataset
        featureList = datasets.Features({'id': datasets.Value('string'),
                                         'context': datasets.Value('string'),
                                         'question': datasets.Value('string'),
                                         'strat_id': datasets.ClassLabel(num_classes=num_strat_classes,
                                                                            names=list(range(num_strat_classes))),
                                         'answers': datasets.Sequence(feature={'text': datasets.Value(dtype='string'),
                                                                               'answer_start': datasets.Value(
                                                                                   dtype='int32')})})


        ## Convert to huggingface dataset
        ds = datasets.DatasetDict()
        # Train
        train_ds = datasets.Dataset.from_pandas(model_input_train, split='train', features=featureList, preserve_index=False)
        ds['train'] = train_ds

        # Split dataset into train/test 60/40 split
        ds = ds["train"].train_test_split(test_size=0.4, stratify_by_column="strat_id", shuffle=True, seed=seed)

        # Split test_set into test/val 50/50 to an overall 60/20/20 split
        # Must do it through intermediary DatasetDict
        val_test_split = ds["test"].train_test_split(test_size=0.5, stratify_by_column="strat_id", shuffle=True,seed=seed)
        ds["test"] = val_test_split["train"]
        ds["val"] = val_test_split["test"]

        # Add negative patellar questions to test set
        ds["test"] = datasets.concatenate_datasets([ds["test"],
                                                    datasets.Dataset.from_pandas(pat_neg, features=featureList,
                                                                        preserve_index=False)])

        # Remove question_id column/feature
        ds = ds.remove_columns("strat_id")
    else:
        # Feature List for dataset
        featureList = datasets.Features({'id': datasets.Value('string'),
                                         'context': datasets.Value('string'),
                                         'question': datasets.Value('string'),
                                         'answers': datasets.Sequence(feature={'text': datasets.Value(dtype='string'),
                                                                               'answer_start': datasets.Value(
                                                                                   dtype='int32')})})

        ## Convert to huggingface dataset
        ds = datasets.DatasetDict()
        # Train
        train_ds = datasets.Dataset.from_pandas(model_input_train, split='train', features=featureList,
                                                preserve_index=False)
        ds['train'] = train_ds

        # Split dataset into train/test 60/40 split
        ds = ds["train"].train_test_split(test_size=0.4, shuffle=True, seed=seed)

        # Split test_set into test/val 50/50 to an overall 60/20/20 split
        # Must do it through intermediary DatasetDict
        val_test_split = ds["test"].train_test_split(test_size=0.5, shuffle=True, seed=seed)
        ds["test"] = val_test_split["train"]
        ds["val"] = val_test_split["test"]

        # Add negative patellar questions to test set
        ds["test"] = datasets.concatenate_datasets([ds["test"],
                                                    datasets.Dataset.from_pandas(pat_neg, features=featureList,
                                                                        preserve_index=False)])

    if trainingDetails["oversample"] == "yes":
        featureList2 = datasets.Features({'id': datasets.Value('string'),
                                         'context': datasets.Value('string'),
                                         'question': datasets.Value('string'),
                                         'answers': datasets.Sequence(
                                             feature={'text': datasets.Value(dtype='string'),
                                                      'answer_start': datasets.Value(dtype='int32')})})

        temp_train = ds["train"].to_pandas()
        temp_train["text"] = temp_train["answers"].apply(lambda x: x["text"][0])
        temp_train["answer_start"] = temp_train["answers"].apply(lambda x: x["answer_start"][0])

        oversample_dict = {}
        for ques in list_ques:
            temp_max = temp_train[temp_train["question"] == ques]["text"].value_counts().max()
            oversample_dict[ques] = temp_train[temp_train["question"] == ques]["text"].value_counts().apply(
                lambda x: temp_max - x).to_dict()

        oversampled_samples = temp_train.groupby("question").apply(lambda x: x.groupby("text").
                                                                   apply(lambda g: g.sample(n=oversample_dict[x.name][g.name],
                                                                                            replace=len(g) < oversample_dict[x.name][g.name])))
        oversampled_samples = oversampled_samples.droplevel([0, 1]).reset_index(drop=True)
        new_train = pd.concat([temp_train, oversampled_samples]).reset_index(drop=True).drop(columns=["text", "answer_start"])
        ds["train"] = datasets.Dataset.from_pandas(new_train, split='train',
                                                   features=featureList2, preserve_index=False)

    # Get total number of samples in test set and number of samples per question
    num_samples_test = {}
    num_samples_test["overall"] = len(ds["test"])
    num_samples_test.update(pd.Series(ds["test"]["question"]).value_counts().to_dict())

    if trainingDetails["model_split"] == "all":
        eval_metrics, pred_ans, act_ans = runModel(outputPath=outputPath, data_ds=ds, ds_dict=ds_dict, list_ques=list_ques,
                                                   modelInfo=modelInfo, trainingDetails=trainingDetails, hyperparameters=hyperparameters)

        print(eval_metrics)

        # Add ground truth labels to predictions
        for i in range(len(pred_ans)):
            pred_ans[i]["actual_text"] = act_ans[i]["answers"]["text"]
            pred_ans[i]["Question"] = ds["test"]["question"][i]

        endTime = datetime.now()
        elapsedTime = endTime - startTime
        printOverallResults(outputPath=outputPath, fileName="OverallResults.csv", modelDetails=modelInfo, dataset_dict = ds_dict,
                            trainingDetails=trainingDetails, hyperparameters=hyperparameters,stats=eval_metrics,
                            predicted_answers=pred_ans, execTime=elapsedTime, list_questions=list_ques,
                            test_num_samples=num_samples_test)

    elif trainingDetails["model_split"] == "one":
        total_pred_ans = []
        temp_pred_ans_for_overall = []
        temp_act_ans_for_overall = []
        ques_metrics = {}
        for curr_question in list_ques:
            ds_sub = ds.filter(lambda row: row["question"]==curr_question)
            eval_metrics, pred_ans, act_ans = runModel(outputPath=outputPath, data_ds=ds_sub, ds_dict=ds_dict,
                                                       list_ques=list_ques,
                                                       modelInfo=modelInfo, trainingDetails=trainingDetails,
                                                       hyperparameters=hyperparameters)
            print(eval_metrics)

            # temp_pred_ans = copy.deepcopy(pred_ans)
            temp_pred_ans_for_overall += copy.deepcopy(pred_ans)
            temp_act_ans_for_overall += copy.deepcopy(act_ans)
            # Add ground truth labels to predictions
            for i in range(len(pred_ans)):
                pred_ans[i]["actual_text"] = act_ans[i]["answers"]["text"]
                pred_ans[i]["Question"] = ds_sub["test"]["question"][i]

            total_pred_ans += pred_ans
            ques_metrics[curr_question] = eval_metrics[curr_question]
            K.clear_session()

        ## Calculate overall metrics
        metric = evaluate.load("squad")
        ques_metrics["overall"] = metric.compute(predictions=temp_pred_ans_for_overall,
                                                 references=temp_act_ans_for_overall)

        endTime = datetime.now()
        elapsedTime = endTime - startTime
        printOverallResults(outputPath=outputPath, fileName="OverallResults.csv", modelDetails=modelInfo,
                            dataset_dict=ds_dict,
                            trainingDetails=trainingDetails, hyperparameters=hyperparameters, stats=ques_metrics,
                            predicted_answers=total_pred_ans, execTime=elapsedTime, list_questions=list_ques,
                            test_num_samples=num_samples_test)


def main():
    if platform.system() == "Windows":
        outputPath = r"D:\zProjects\QA\results"
        # outputPath = r"C:\Users\David Lee\Desktop\TKA"


    elif platform.system() == "Linux":
        outputPath = r"/home/dmlee/QA/results"


    outputPath = os.path.join(outputPath, "2023-05-03")
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # Models to choose from (+ indicates cased versions ^ indicates uncased), * indicates not currently working)
    # distilbert-base-uncased, distilbert-base-cased
    # distilbert-base-uncased-distilled-squad, distilbert-base-cased-distilled-squad
    # bert-base-uncased, bert-base-cased
    # dmis-lab/biobert-v1.1
    # emilyalsentzer/Bio_ClinicalBERT
    # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    # roberta-base, roberta-large
    # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    # decapoda-research/llama-7b-hf
    # xlnet-base-cased, xlnet-large-cased
    # flan-t5-base, flan-t5-small, flan-t5-large, flan-t5-xl, flan-t5-xxl (*)

    modelDetails = {"name":"distilbert-base-uncased-distilled-squad"}

    trainDetails = {"type":"split",        # 'split' for train/val/test split
                    "model_split": "all",  # 'all' if one model, 'one' if seperate model for each question
                    "strat_on":"none",     # 'answers' if strat on answers, 'questions' for questions, 'none' for no strat
                    "oversample":"no",     # 'yes' if oversampling, 'no' for no oversampling
                    "modelType":"custom",        #'custom' if using custom implementation, otherwise generic HF implementation
                    "notes":""}        # '' if no notes otherwise add notes

    dataDetails = {"num_questions": 4,           # '3' if using original three, '4' if including patella question
                   "Use_prebuilt_labels": "no",  # 'yes' to use prebuilt data and modified dataset, 'no' to perform that as part of pipeline
                   "scale_back": 2}              # '0' for no scale back, '1' for removing ps/cr unabbreviation,
                                                 # '2' for original files without changing any of the OP Notes

    hyperparameters = {"epochs": 2,
                       "max_length": 384,
                       "doc_stride": 128,
                       "batch_size": 16,
                       "learning_rate":3e-5}

    ds_dict = {"train":"smaller"}

    def runLoop():
        modelsList = ["distilbert", "bert"]
        caseList = ["lowercase", "uppercase"]
        for i in range(1, 5):
            hyperparameters["epochs"] = i
            for modelToRun in modelsList:
                for caseToRun in caseList:
                    modelDetails["name"] = modelToRun
                    modelDetails["case"] = caseToRun
                    if modelToRun == "bert":
                        hyperparameters["batch_size"] = 4
                    elif modelToRun == "distilbert":
                        hyperparameters["batch_size"] = 16
                    Pipeline(outputPath, ds_dict, modelDetails, trainDetails, hyperparameters)
                    K.clear_session()


    Pipeline(outputPath=outputPath, ds_dict=ds_dict, modelInfo=modelDetails, dataDetails=dataDetails,
             trainingDetails=trainDetails, hyperparameters=hyperparameters)
    K.clear_session()


#runLoop()


if platform.system() == "Windows":
    dataPath = r"C:\Users\dmlee\PycharmProjects\TKA"
    # dataPath = r"C:\Users\David Lee\Desktop\TKA"

elif platform.system() == "Linux":
    dataPath = r"/home/dmlee/TKA"

main()
