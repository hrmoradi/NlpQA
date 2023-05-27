import platform

import torch
from Libraries import *
from HelperFunctions import importCustomModel, compute_metrics, compute_metrics2
from ModelFunctions import myDistilBertForQuestionAnswering

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, get_peft_model_state_dict, set_peft_model_state_dict
import bitsandbytes as bnb
import accelerate
from accelerate import Accelerator
from transformers import DefaultDataCollator

def reg_torch(modelInfo, tokenizer, train_set, val_set_init, val_set, test_set, validation_dataset, hyperparameters, data_ds, list_ques):

    model = importCustomModel(modelInfo["name"])
    model.model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_set = train_set.remove_columns("token_type_ids")
    train_ds = train_set.with_format("torch", device=device)

    val_set = val_set.remove_columns("token_type_ids")
    val_ds = val_set.with_format("torch", device=device)

    test_set = test_set.remove_columns("token_type_ids")
    test_ds = test_set.with_format("torch", device=device)

    train_dataloader = DataLoader(train_ds, batch_size=hyperparameters["batch_size"])
    val_dataloader = DataLoader(val_ds, batch_size=hyperparameters["batch_size"])
    test_dataloader = DataLoader(test_ds, batch_size=hyperparameters["batch_size"])


    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = hyperparameters["epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            with torch.cuda.amp.autocast():
                outputs = model(**batch, return_dict=True)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            progress_bar.update(1)

        model.eval()
        start_logits = []
        end_logits = []
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        if len(list_ques) == 4:
            val_metrics, val_pred_ans, val_act_ans = compute_metrics2(start_logits, end_logits, val_set_init,
                                                                      data_ds["val"], list_ques)
        else:
            val_metrics, val_pred_ans, val_act_ans = compute_metrics(start_logits, end_logits, val_set_init,
                                                                     data_ds["val"],
                                                                     list_ques)
        print(f"epoch {epoch}:", val_metrics)

    model.eval()
    all_start_logits = []
    all_end_logits = []
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch, return_dict=True)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_set, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_set, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    eval_metrics, pred_ans, act_ans = compute_metrics2(start_logits_concat, end_logits_concat, validation_dataset,
                                                       data_ds["test"], list_ques)

    return eval_metrics, pred_ans, act_ans


def acc_torch(modelInfo, tokenizer, train_set, val_set_init, val_set, test_set, validation_dataset, hyperparameters, data_ds, list_ques):

    train_set = train_set.remove_columns("token_type_ids")
    train_ds = train_set.with_format("torch")

    val_set = val_set.remove_columns("token_type_ids")
    val_ds = val_set.with_format("torch")

    test_set = test_set.remove_columns("token_type_ids")
    test_ds = test_set.with_format("torch")

    train_dataloader = DataLoader(train_ds, batch_size=hyperparameters["batch_size"])
    val_dataloader = DataLoader(val_ds, batch_size=hyperparameters["batch_size"])
    test_dataloader = DataLoader(test_ds, batch_size=hyperparameters["batch_size"])

    model = importCustomModel(modelInfo["name"])
    model.model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    accelerator = Accelerator(fp16=True)

    model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader
    )

    num_train_epochs = hyperparameters["epochs"]
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, return_dict=True)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        accelerator.print("Evaluation!")
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch, return_dict=True)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        if len(list_ques) == 4:
            val_metrics, val_pred_ans, val_act_ans = compute_metrics2(start_logits, end_logits, val_set_init,
                                                                      data_ds["val"], list_ques)
        else:
            val_metrics, val_pred_ans, val_act_ans = compute_metrics(start_logits, end_logits, val_set_init,
                                                                     data_ds["val"], list_ques)
        print(f"epoch {epoch}:", val_metrics["overall"])



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=T-gy-LxM0yAi
def peft_torch(modelInfo, tokenizer, train_set, val_set_init, val_set, test_set, validation_dataset, hyperparameters, data_ds, list_ques):
    if platform.system() == "Windows":
        output_dir = r"D:\zProjects\[Models]\llama-8bit"
    elif platform.system() == "Linux":
        # output_dir = r"/home/dmlee/[models]/llama-8bit"
        output_dir = r"/home/dmlee892/models/llama-8bit-small"
        output_dir2 = r"/home/dmlee892/models/llama-8bit-small2"

        resume_from_checkpoint = r"/home/dmlee892/models/llama-8bit-small"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(output_dir2)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    train_set = train_set.remove_columns("token_type_ids")
    train_ds = train_set.with_format("torch", device=device)

    val_set = val_set.remove_columns("token_type_ids")
    val_ds = val_set.with_format("torch", device=device)

    test_set = test_set.remove_columns("token_type_ids")
    test_ds = test_set.with_format("torch", device=device)

    # train_dataloader = DataLoader(train_ds, batch_size=hyperparameters["batch_size"])
    # val_dataloader = DataLoader(val_ds, batch_size=hyperparameters["batch_size"])
    test_dataloader = DataLoader(test_ds, batch_size=hyperparameters["batch_size"])

    model = importCustomModel(modelInfo["name"])
    # model = myDistilBertForQuestionAnswering(modelInfo["name"])
    model.model.resize_token_embeddings(len(tokenizer))

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.model.enable_input_require_grads()

    # Values taken from
    # https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L182
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model.model = prepare_model_for_int8_training(model.model)
    model.model = get_peft_model(model.model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    print_trainable_parameters(model)

    micro_batch_size = 1
    val_set_size = 2000
    gradient_accumulation_steps = hyperparameters["batch_size"] // micro_batch_size
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=hyperparameters["epochs"],
            learning_rate=hyperparameters["learning_rate"],
            fp16=True,
            logging_steps=100,
            optim="adamw_torch",
            prediction_loss_only=True,
            evaluation_strategy="epoch",
            eval_accumulation_steps=hyperparameters["batch_size"],
            save_strategy="epoch",
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=200,
            load_best_model_at_end=True if val_set_size > 0 else False,
            label_names=["start_positions", "end_positions"],
            remove_unused_columns=False,
        ),
        data_collator=DefaultDataCollator(),
        #data_collator=transformers.DataCollatorForSeq2Seq(
        #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #),
    )


    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    #
    old_state_dict = model.state_dict
    old_state_dict2 = model.model.state_dict
    # model.model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model.model, type(model.model))

    model.to(device)

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # with torch.cuda.amp.autocast():
    # trainer.train()

    model.save_pretrained(output_dir, state_dict=old_state_dict())
    model.model.save_pretrained(output_dir2, state_dict=old_state_dict2())

    model.config.use_cache = True

    all_start_logits = []
    all_end_logits = []
    for batch in test_dataloader:
        with torch.no_grad():
            _, predictions, _ = trainer.prediction_step(model=model,
                                                        inputs=batch,
                                                        prediction_loss_only=False)

            all_start_logits.append(predictions[0].cpu().numpy())
            all_end_logits.append(predictions[1].cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits = np.concatenate(all_end_logits)

    all_start_logits = all_start_logits[: len(validation_dataset)]
    all_end_logits = all_end_logits[: len(validation_dataset)]

    # predictions, _, _ = trainer.predict(test_ds)
    # predictions, _, _ = trainer.prediction_loop(test_dataloader, description="Prediction", prediction_loss_only=False)


    #

    if len(list_ques) == 4:
        eval_metrics, pred_ans, act_ans = compute_metrics2(all_start_logits, all_end_logits, validation_dataset,
                                                                  data_ds["test"], list_ques)
    else:
        eval_metrics, pred_ans, act_ans = compute_metrics(all_start_logits, all_end_logits, validation_dataset,
                                                                 data_ds["test"], list_ques)

    return eval_metrics, pred_ans, act_ans