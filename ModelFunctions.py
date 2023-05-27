from Libraries import *

# Currently working on adding custom model class based on transformers
# However, currently returning much worse results despite it
# supposed to be identical copy right now...

class MyTFQuestionAnswering(TFPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, modelName, pathName, *inputs, **kwargs):
        models_with_no_tf = ['dmis-lab/biobert-v1.1', 'milyalsentzer/Bio_ClinicalBERT',
                             'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext']

        config = AutoConfig.from_pretrained(pathName)
        super().__init__(config, *inputs, **kwargs)

        self.modelName = modelName

        self.model = TFAutoModel.from_pretrained(pathName, config=config, name="main_model",
                                                 from_pt=modelName in models_with_no_tf)

        # T5Config does not support initializer_range
        if "t5" not in modelName.lower():
            self.qa_outputs = tf.keras.layers.Dense(
                config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
            )
        # T5Config supports initializer_factor
        elif "t5" in modelName.lower():
            self.qa_outputs = tf.keras.layers.Dense(
                config.num_labels, kernel_initializer=get_initializer(config.initializer_factor), name="qa_outputs"
            )

        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"

        if self.modelName == "distilbert":
            self.dropout = tf.keras.layers.Dropout(config.qa_dropout)

    @unpack_inputs
    def call(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            start_positions=None,
            end_positions=None,
            training=False,
    ):
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        hidden_states = model_output[0]  # (bs, max_query_len, dim)

        if self.modelName == "distilbert":
            hidden_states = self.dropout(hidden_states, training=training)  # (bs, max_query_len, dim)

        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + model_output[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForQuestionAnswering.serving_output
    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits, end_logits=output.end_logits, hidden_states=hs, attentions=attns
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForQuestionAnswering.serving_output
    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits, end_logits=output.end_logits, hidden_states=hs, attentions=attns
        )


class MyLlamaTFQuestionAnswering(TFPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, modelName, pathName, *inputs, **kwargs):
        models_with_no_tf = ['decapoda-research/llama-7b-hf']

        config = AutoConfig.from_pretrained(pathName)

        super().__init__(config, *inputs, **kwargs)

        self.modelName = modelName
        self.model = TFLlamaModel.from_pretrained(pathName, config=config, from_pt=True, name="main_model")

        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            start_positions=None,
            end_positions=None,
            training=False,
    ):
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        model_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = model_output[0]  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + model_output[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )


class MyLlamaQuestionAnswering(PreTrainedModel):
    def __init__(self, modelName, pathName, *inputs, **kwargs):

        config = AutoConfig.from_pretrained(pathName)
        super().__init__(config, *inputs, **kwargs)

        self.modelName = modelName

        self.model = AutoModel.from_pretrained(pathName, config=config, load_in_8bit=True, device_map="auto")

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        if config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            start_positions=None,
            end_positions=None,
    ):
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = model_output[0]  # (bs, max_query_len, dim)

        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + model_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )

class MyXLnetTFQuestionAnswering(TFXLNetPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, modelName, pathName, *inputs, **kwargs):
        # Auto functions like AutoConfig, AutoModel, etc. do not currently support Llama
        config = AutoConfig.from_pretrained(pathName)
        super().__init__(config, *inputs, **kwargs)

        self.modelName = modelName
        self.model = TFAutoModel.from_pretrained(pathName, config=config)

        self.qa_outputs = tf.keras.layers.Dense(
                config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
            )

        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"

    @unpack_inputs
    def call(
            self,
            input_ids = None,
            attention_mask = None,
            mems = None,
            perm_mask = None,
            target_mapping = None,
            token_type_ids = None,
            input_mask = None,
            head_mask = None,
            inputs_embeds = None,
            use_mems = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            start_positions = None,
            end_positions = None,
            training = False,
    ):
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        hidden_states = model_output[0]  # (bs, max_query_len, dim)

        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + model_output[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetForQuestionAnsweringSimpleOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            mems=model_output.mems,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        mems = tf.convert_to_tensor(output.mems) if output.mems is not None else None

        return TFXLNetForQuestionAnsweringSimpleOutput(
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            mems=mems,
            hidden_states=hs,
            attentions=attns,
        )

from transformers import DistilBertPreTrainedModel, DistilBertModel
class myDistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, modelName):
        config = AutoConfig.from_pretrained(modelName)
        super().__init__(config)

        # self.distilbert = DistilBertModel.from_pretrained(modelName, torch_dtype=torch.float16)
        self.model = DistilBertModel.from_pretrained(modelName)
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        if config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")

        self.dropout = nn.Dropout(config.qa_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.model.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.model.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        start_positions = None,
        end_positions = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ) :
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )