from Libraries import *

from HelperFunctions import ReturnNotes, extractModelInfo, addQuestion, importModelandTokenizer, preprocess_function,\
    preprocess_validation_examples, compute_metrics, printOverallResults


def Pipeline(outputPath, ds_dict, modelInfo, trainingDetails, hyperparameters):
    startTime = datetime.now()

    # Get train set
    medicalNotes_train = ReturnNotes(ds_dict["train"], dataPath)
    medicalNotes_train = medicalNotes_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    model_input_train = extractModelInfo(medicalNotes_train)

    # # Get test set
    # medicalNotes_test = ReturnNotes(ds_dict["test"])
    # medicalNotes_test = medicalNotes_test.sample(frac=1, random_state=seed).reset_index(drop=True)
    # model_input_test = extractModelInfo(medicalNotes_test)

    if modelInfo["case"] == "lowercase":
        model_input_train[["question", "context", "text"]] = model_input_train[["question", "context", "text"]].astype(str).apply(lambda col: col.str.lower())
        # model_input_test[["question", "context", "text"]] = model_input_test[["question", "context", "text"]].astype(str).apply(lambda col: col.str.lower())

    # Convert to correct nesting
    model_input_train["answers"] = model_input_train.apply(lambda x: {"text": [x["text"]], "answer_start": [x["answer_start"]]},
                                                           axis=1)

    # model_input_test["answers"] = model_input_test.apply(lambda x: {"text": [x["text"]], "answer_start": [x["answer_start"]]},
    #                                                      axis=1)

    # Drop unncessary columns
    model_input_train = model_input_train.drop(["text", "answer_start"], axis=1)
    # model_input_test = model_input_test.drop(["text", "answer_start"], axis=1)

    # Create new column question_id that maps to an integer to stratify over
    list_ques = model_input_train["question"].unique().tolist()
    num_ques = len(list_ques)
    ques_conv_dict = {list_ques[ques_idx]: ques_idx for ques_idx in range(len(list_ques))}
    ques_conv_dict

    model_input_train["question_id"] = model_input_train["question"].apply(lambda x: ques_conv_dict[x])
    model_input_train

    # # Include only these questions
    # model_input_train = model_input_train[(model_input_train.question == "who is the maker of the implant?")]
    # model_input_test = model_input_test[(model_input_test.question == "who is the maker of the implant?")]

    # Feature List for dataset
    featureList = datasets.Features({'id': datasets.Value('string'),
                                     'context': datasets.Value('string'),
                                     'question': datasets.Value('string'),
                                     'question_id': datasets.ClassLabel(num_classes=num_ques,
                                                                        names=list(range(num_ques))),
                                     'answers': datasets.Sequence(feature={'text': datasets.Value(dtype='string'),
                                                                           'answer_start': datasets.Value(
                                                                               dtype='int32')})})


    ## Convert to huggingface dataset
    ds = datasets.DatasetDict()
    # Train
    train_ds = datasets.Dataset.from_pandas(model_input_train, split='train', features=featureList, preserve_index=False)
    ds['train'] = train_ds

    # Split dataset into train/test 60/40 split
    ds = ds["train"].train_test_split(test_size=0.4, stratify_by_column="question_id", shuffle=True, seed=seed)

    # Split test_set into test/val 50/50 to an overall 60/20/20 split
    # Must do it through intermediary DatasetDict
    val_test_split = ds["test"].train_test_split(test_size=0.5, stratify_by_column="question_id", shuffle=True,seed=seed)
    ds["test"] = val_test_split["train"]
    ds["val"] = val_test_split["test"]

    # Remove question_id column/feature
    ds = ds.remove_columns("question_id")

    # Import tokenizer and model
    tokenizer, model = importModelandTokenizer(modelInfo["name"], modelInfo["case"])

    # The maximum length of a feature (question and context)
    max_length = hyperparameters["max_length"]
    # The authorized overlap between two part of the context when splitting
    doc_stride = hyperparameters["doc_stride"]

    # Tokenize inputs for training
    tokenized_dataset = ds.map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=ds["train"].column_names)

    # Convert to format useable with tensorflow

    train_set = tokenized_dataset["train"].with_format("numpy")[:]
    val_set = tokenized_dataset["val"].with_format("numpy")[:]
    # test_set = tokenized_dataset["test"].with_format("numpy")[:]

    # Tokenize inputs for evaluation set
    validation_dataset = ds.map(
        preprocess_validation_examples,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=ds['test'].column_names,
    )

    test_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    test_set = test_set["test"].with_format("numpy")[:]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_set)
    train_dataset = train_dataset.batch(hyperparameters["batch_size"])

    val_dataset = tf.data.Dataset.from_tensor_slices(val_set)
    val_dataset = val_dataset.batch(hyperparameters["batch_size"])

    test_dataset = tf.data.Dataset.from_tensor_slices(test_set)
    test_dataset = test_dataset.batch(hyperparameters["batch_size"])

    ## Use below if using GPU, otherwise leave commented out
    # keras.mixed_precision.set_global_policy("mixed_float16")

    optimizer = keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])
    model.compile(optimizer=optimizer)

    model.fit(train_dataset, validation_data=val_dataset, epochs=hyperparameters["epochs"])
    # model.fit(train_dataset, epochs=hyperparameters["epochs"])

    # Get starting and ending logits
    outputs = model.predict(test_dataset)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Evaluate test
    eval_metrics, pred_ans, act_ans = compute_metrics(start_logits, end_logits, validation_dataset["test"], ds["test"])
    print(eval_metrics)

    # Add ground truth labels to predictions
    for i in range(len(pred_ans)):
        pred_ans[i]["actual_text"] = act_ans[i]["answers"]["text"]
        pred_ans[i]["Question"] = ds["test"]["question"][i]

    endTime = datetime.now()
    elapsedTime = endTime - startTime
    printOverallResults(outputPath=outputPath, fileName="OverallResults.csv", modelDetails=modelInfo, dataset_dict = ds_dict,
                        trainingDetails=trainingDetails, hyperparameters=hyperparameters,stats=eval_metrics,
                        predicted_answers=pred_ans, execTime=elapsedTime)



def main():
    if platform.system() == "Windows":
        outputPath = r"D:\zProjects\QA\results"

    elif platform.system() == "Linux":
        outputPath = r"/home/dmlee/QA/results"


    fileType = "larger"
    modelDetails = {"name":"distilbert",
                    "case":"lowercase"}

    trainDetails = {"type":"split"}

    hyperparameters = {"epochs": 5,
                       "max_length": 384,
                       "doc_stride": 128,
                       "batch_size": 32,
                       "learning_rate":5e-5}

    ds_dict = {"train":"larger",
               "test":"smaller"}

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


if platform.system() == "Windows":
    dataPath = r"C:\Users\dmlee\PycharmProjects\TKA"

elif platform.system() == "Linux":
    dataPath = r"/home/dmlee/TKA"

seed = 42
main()
