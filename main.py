from Libraries import *
from transformers import TFAutoModel
from HelperFunctions import ReturnNotes, extractModelInfo, changeOPNote, importModelandTokenizer, importTokenizer, \
    preprocess_function, preprocess_validation_examples, compute_metrics, printOverallResults

from ModelFunctions import MyTFQuestionAnswering


def runModel(outputPath, data_ds, ds_dict, list_ques, modelInfo, trainingDetails, hyperparameters):
    # Import tokenizer and model
    # tokenizer, model = importModelandTokenizer(modelInfo["name"], modelInfo["case"])
    tokenizer = importTokenizer(modelInfo["name"], modelInfo["case"])


    # The maximum length of a feature (question and context)
    max_length = hyperparameters["max_length"]
    # The authorized overlap between two part of the context when splitting
    doc_stride = hyperparameters["doc_stride"]

    # Tokenize inputs for training
    tokenized_dataset = data_ds.map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=data_ds["train"].column_names)

    # Convert to format useable with tensorflow

    # train_set = tokenized_dataset["train"].to_tf_dataset(
    #     columns=["input_ids", "attention_mask"],
    #     label_cols=["start_positions", "end_positions"],
    #     batch_size=hyperparameters["batch_size"],
    #     shuffle=False)

    train_set = tokenized_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
        batch_size=hyperparameters["batch_size"],
        shuffle=False)

    val_set = val_set = tokenized_dataset["val"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["start_positions", "end_positions"],
        batch_size=hyperparameters["batch_size"],
        shuffle=False)
    # test_set = tokenized_dataset["test"].with_format("numpy")[:]

    # Tokenize inputs for evaluation set
    validation_dataset = data_ds.map(
        preprocess_validation_examples,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=data_ds['test'].column_names,
    )

    test_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    test_set = tokenized_dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["start_positions", "end_positions"],
        batch_size=hyperparameters["batch_size"],
        shuffle=False)

    ## Use below if using GPU, otherwise leave commented out
    # keras.mixed_precision.set_global_policy("mixed_float16")

    modelName = "distilbert-base-uncased-distilled-squad"
    # configN = AutoConfig.from_pretrained(modelName)
    model = MyTFQuestionAnswering(modelName)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])
    model.compile(optimizer=optimizer)
    model.fit(train_set, epochs=1)

    # Get starting and ending logits
    outputs = model.predict(test_set)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Evaluate test
    eval_metrics, pred_ans, act_ans = compute_metrics(start_logits, end_logits, validation_dataset["test"], data_ds["test"],
                                                      list_ques)

    return [eval_metrics, pred_ans, act_ans]

def Pipeline(outputPath, ds_dict, modelInfo, trainingDetails, hyperparameters):
    startTime = datetime.now()

    # Get train set
    medicalNotes_train = ReturnNotes(ds_dict["train"], dataPath)

    # Modify pat_id to be concatenation of pat_id and CPT CODE
    # since patient can have more than OP Note associated with them
    medicalNotes_train["pat_id"] = medicalNotes_train.apply(
        lambda x: x["pat_id"] + "_" + x["CPT Code Date"].strftime("%Y%m%d"), axis=1)

    # Change appropriate strings such as 'cr' to 'cruciate retaining' to match labels
    medicalNotes_train["OP_NOTE"] = medicalNotes_train["OP_NOTE"].apply(lambda x: changeOPNote(x))
    medicalNotes_train = medicalNotes_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    model_input_train = extractModelInfo(medicalNotes_train)

    if modelInfo["case"] == "lowercase":
        model_input_train[["question", "context", "text"]] = model_input_train[["question", "context", "text"]].astype(str).apply(lambda col: col.str.lower())

    # Convert to correct nesting
    model_input_train["answers"] = model_input_train.apply(lambda x: {"text": [x["text"]], "answer_start": [x["answer_start"]]},
                                                           axis=1)
    # Drop unncessary columns
    model_input_train = model_input_train.drop(["text", "answer_start"], axis=1)

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

    elif platform.system() == "Linux":
        outputPath = r"/home/dmlee/QA/results"


    outputPath = os.path.join(outputPath, "28_03_23")
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)


    modelDetails = {"name":"distilbert",
                    "case":"lowercase"}

    trainDetails = {"type":"split",       # 'split' for train/val/test split
                    "model_split": "all", # 'all' if one model, 'one' if seperate model for each question
                    "strat_on":"none", # 'answers' if strat on answers, 'questions' for questions, 'none' for no strat
                    "oversample":"no"} # 'yes' if oversampling, 'no' for no oversampling

    hyperparameters = {"epochs": 1,
                       "max_length": 384,
                       "doc_stride": 128,
                       "batch_size": 16,
                       "learning_rate":3e-5}

    ds_dict = {"train":"smaller",
               "test":"smaller"}

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


    Pipeline(outputPath, ds_dict, modelDetails, trainDetails, hyperparameters)
    K.clear_session()

    #runLoop()


if platform.system() == "Windows":
    dataPath = r"C:\Users\dmlee\PycharmProjects\TKA"

elif platform.system() == "Linux":
    dataPath = r"/home/dmlee/TKA"

main()
