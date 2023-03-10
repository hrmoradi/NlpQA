from Libraries import *

from HelperFunctions import ReturnNotes, extractModelInfo, addQuestion, importModelandTokenizer, preprocess_function,\
    preprocess_validation_examples, compute_metrics


def Pipeline(ds_dict, hyperparameters):
    medicalNotes_train = ReturnNotes(ds_dict["train"])
    medicalNotes_train = medicalNotes_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    model_input_train = extractModelInfo(medicalNotes_train)

    # Convert to correct nesting
    model_input_train["answers"] = model_input_train.apply(lambda x: {"text": [x["text"]], "answer_start": [x["answer_start"]]},
                                               axis=1)

    model_input_train = model_input_train.drop(["text", "answer_start"], axis=1)

    # Get test set aka smaller set
    medicalNotes_test = ReturnNotes(ds_dict["test"])
    medicalNotes_test = medicalNotes_test.sample(frac=1, random_state=seed).reset_index(drop=True)
    model_input_test = extractModelInfo(medicalNotes_test)

    # Convert to correct nesting
    model_input_test["answers"] = model_input_test.apply(
        lambda x: {"text": [x["text"]], "answer_start": [x["answer_start"]]},
        axis=1)

    model_input_test = model_input_test.drop(["text", "answer_start"], axis=1)

    # Include only these questions
    model_input_train = model_input_train[(model_input_train.question == "Who is the maker of the implant?")]
    model_input_test = model_input_test[(model_input_test.question == "Who is the maker of the implant?")]

    # Feature List for dataset
    featureList = datasets.Features({'id': datasets.Value('string'),
                                     'context': datasets.Value('string'),
                                     'question': datasets.Value('string'),
                                     'answers': datasets.Sequence(feature={'text': datasets.Value(dtype='string'),
                                                                           'answer_start': datasets.Value(dtype='int32')})})


    ## Convert to huggingface dataset
    ds = datasets.DatasetDict()
    # Train
    train_ds = datasets.Dataset.from_pandas(model_input_train, split='train', features=featureList, preserve_index=False)
    ds['train'] = train_ds

    # Test
    test_ds = datasets.Dataset.from_pandas(model_input_test, split='test', features=featureList, preserve_index=False)
    ds['test'] = test_ds


    tokenizer, model = importModelandTokenizer("DistilBert")

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

    # Tokenize inputs for evaluation set
    validation_dataset = ds.map(
        preprocess_validation_examples,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'doc_stride': doc_stride},
        batched=True,
        remove_columns=ds['test'].column_names,
    )

    val_set = validation_dataset["test"].remove_columns(["example_id", "offset_mapping"])
    val_set = val_set.with_format("numpy")[:]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_set)
    train_dataset = train_dataset.batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_set)
    val_dataset = val_dataset.batch(32)

    ## Use below if using GPU, otherwise leave commented out
    # keras.mixed_precision.set_global_policy("mixed_float16")

    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)

    # model.fit(train_set, validation_data=validation_set, epochs=1)
    model.fit(train_dataset, epochs=hyperparameters["epochs"])

    # Get starting and ending logits
    outputs = model.predict(val_dataset)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Evaluate test
    eval_metrics, pred_ans, act_ans = compute_metrics(start_logits, end_logits, validation_dataset["test"], ds["test"])
    print(eval_metrics)

    # Add ground truth labels to predictions
    for i in range(len(pred_ans)):
        pred_ans[i]["actual_text"] = act_ans[i]["answers"]["text"]
        pred_ans[i]["Question"] = ds["train"]["question"][i]


    if hyperparameters["epochs"] == 1:
        outName = f'Predicted Output [New] - {hyperparameters["epochs"]} Epoch.txt'
    else:
        outName = f'Predicted Output [New] - {hyperparameters["epochs"]} Epochs.txt'

    with open(outName, 'w') as f:
        f.write(f"{eval_metrics}\n\n")
        for line in pred_ans:
            f.write(f"{line}\n")


def main():
    fileType = "larger"
    hyperparameters = {"epochs": 1,
                       "max_length": 384,
                       "doc_stride": 128}

    ds_dict = {"train":"larger",
               "test":"smaller"}
    for i in range(3,6):
        hyperparameters["epochs"] = i
        Pipeline(ds_dict, hyperparameters)

seed = 42
main()