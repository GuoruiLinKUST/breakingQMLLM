from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.models.llava.configuration_llava import LlavaConfig
import os
import torch
import argparse
from PIL import Image
from utils.config import Args
from models.unitok import UniTok
from utils.process import normalize_01_into_pm1
from torchvision.transforms import transforms, InterpolationMode
import warnings
import torch.nn.functional as F


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"

_CHECKPOINT_FOR_DOC = "llava-hf/llava-1.5-7b-hf"


@dataclass
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class VectorQuantizerCLS(nn.Module):
    """
    Vector quantizer with classification capability for image content filtering.

    This module performs vector quantization on input embeddings and can classify them
    into predefined categories using a learned codebook.

    Args:
        num_embeddings (`int`, defaults to 128):
            Size of the codebook (number of discrete codes).
        embedding_dim (`int`, defaults to 4096):
            Dimensionality of the embeddings.
        commitment_cost (`float`, defaults to 0.25):
            Weight for the commitment loss term.
        codebook_path (`str`, *optional*):
            Path to load a pre-trained codebook.
        mapping_path (`str`, *optional*):
            Path to load category mapping information.
        use_cosine (`bool`, defaults to True):
            Whether to use cosine similarity for distance computation.
        randomize_indices (`bool`, defaults to True):
            Whether to randomize codebook indices during initialization.
    """
    def __init__(self, num_embeddings: int = 128, embedding_dim: int = 4096, commitment_cost: float = 0.25,
                 codebook_path: str = None, mapping_path: str = None, use_cosine: bool = True,
                 randomize_indices: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_cosine = use_cosine

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        self.register_buffer('_category_mapping_indices', torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer('_category_mapping_names', torch.zeros(num_embeddings, dtype=torch.long))

        self.center_to_category = None

    def _store_category_mapping(self):
        """
        Store the category mapping information as model buffers for persistence.

        This method saves the mapping between codebook indices and their corresponding
        category IDs, allowing the mapping to be saved and loaded with model checkpoints.
        """
        if not self.center_to_category:
            warnings.warn("No category mapping to store")
            return

        all_categories = sorted(set(self.center_to_category.values()))

        indices = list(self.center_to_category.keys())
        category_ids = [self.center_to_category[idx] for idx in indices]

        if len(indices) != self._category_mapping_indices.size(0):
            self.register_buffer('_category_mapping_indices', torch.zeros(len(indices), dtype=torch.long))
            self.register_buffer('_category_mapping_names', torch.zeros(len(indices), dtype=torch.long))

        self._category_mapping_indices.copy_(torch.tensor(indices, dtype=torch.long))
        self._category_mapping_names.copy_(torch.tensor(category_ids, dtype=torch.long))

        print(f"Stored category mapping with {len(indices)} entries and {len(all_categories)} unique categories")

    def _load_category_mapping(self):
        """
        Load the category mapping information from model buffers.

        Returns:
            mapping (`dict`): Dictionary mapping codebook indices to category IDs.
        """
        if not hasattr(self, '_category_mapping_indices') or self._category_mapping_indices.numel() == 0:
            warnings.warn("No stored category mapping found")
            return {}

        indices = self._category_mapping_indices.tolist()
        category_ids = self._category_mapping_names.tolist()

        mapping = {}
        for idx, cat_id in zip(indices, category_ids):
            mapping[idx] = cat_id

        return mapping

    def forward(self, inputs):
        """
        Forward pass for vector quantization with classification.

        Args:
            inputs (`torch.FloatTensor`):
                Input embeddings of shape `(batch_size, embedding_dim)`.

        Returns:
            quantized (`torch.FloatTensor`): Quantized embeddings.
            encoding_indices (`torch.LongTensor`): Indices of the selected codebook entries.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected input shape (batch_size, {self.embedding_dim}), got {inputs.shape}")

        self.embedding.weight.data = self.embedding.weight.data.to(dtype=inputs.dtype)

        flat_input = inputs

        if self.use_cosine:
            normalized_input = F.normalize(flat_input, p=2, dim=1)
            normalized_weights = F.normalize(self.embedding.weight, p=2, dim=1)

            cosine_sim = torch.matmul(normalized_input, normalized_weights.t())

            distances = 1 - cosine_sim
        else:
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                         + torch.sum(self.embedding.weight**2, dim=1)
                         - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device).to(inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight)

        quantized = flat_input + (quantized - flat_input).detach()

        return quantized, encoding_indices.squeeze()

    def encode(self, inputs):
        """
        Encode input embeddings to their nearest codebook indices.

        Args:
            inputs (`torch.FloatTensor`):
                Input embeddings of shape `(batch_size, embedding_dim)`.

        Returns:
            encoding_indices (`torch.LongTensor`): Indices of the nearest codebook entries.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected input shape (batch_size, {self.embedding_dim}), got {inputs.shape}")

        with torch.no_grad():
            if self.use_cosine:
                normalized_input = F.normalize(inputs, p=2, dim=1)
                normalized_weights = F.normalize(self.embedding.weight, p=2, dim=1)
                cosine_sim = torch.matmul(normalized_input, normalized_weights.t())
                distances = 1 - cosine_sim
            else:
                distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                             + torch.sum(self.embedding.weight**2, dim=1)
                             - 2 * torch.matmul(inputs, self.embedding.weight.t()))

            encoding_indices = torch.argmin(distances, dim=1)

            return encoding_indices

    def get_category_from_index(self, indices):
        """
        Map codebook indices to their corresponding category IDs.

        Args:
            indices (`torch.LongTensor`):
                Codebook indices to map.

        Returns:
            categories (`list`): List of category IDs corresponding to the input indices.
        """
        if self.center_to_category is None:
            self.center_to_category = self._load_category_mapping()

        if not self.center_to_category:
            return [0] * indices.numel()

        indices_np = indices.cpu().numpy().flatten()

        categories = []
        for idx in indices_np:
            idx_int = int(idx)
            category = self.center_to_category.get(idx_int, 0)
            categories.append(category)

        return categories

    def classify(self, inputs):
        """
        Classify input embeddings into categories.

        Args:
            inputs (`torch.FloatTensor`):
                Input embeddings of shape `(batch_size, embedding_dim)`.

        Returns:
            categories (`list`): List of predicted category IDs.
            indices (`torch.LongTensor`): Corresponding codebook indices.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected input shape (batch_size, {self.embedding_dim}), got {inputs.shape}")

        indices = self.encode(inputs)
        categories = self.get_category_from_index(indices)
        return categories, indices


class LlavaMultiModalProjector(nn.Module):
    """
    Multi-modal projector for mapping vision features to language model space.

    This module projects vision encoder outputs into the text embedding space and
    includes a vector quantizer for content classification.

    Args:
        config (`LlavaConfig`):
            Model configuration containing projection and vision settings.
    """
    def __init__(self, config: LlavaConfig):
        super().__init__()
        print(config)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

        self.vq_cls = VectorQuantizerCLS(
            num_embeddings=128,
            embedding_dim=config.text_config.hidden_size,
            commitment_cost=0.25,
            use_cosine=True
        )

    def forward(self, image_features):
        """
        Project image features and perform content classification.

        Args:
            image_features (`torch.FloatTensor`):
                Vision encoder outputs.

        Returns:
            hidden_states (`torch.FloatTensor`): Projected features.
            indices (`torch.LongTensor`): Classification codebook indices.
            categories (`list`): Predicted category IDs.

        Raises:
            ValueError: If harmful content (non-zero category) is detected.
        """
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        cls_features = hidden_states.mean(dim=1)

        quantized, indices = self.vq_cls(cls_features)
        categories = self.vq_cls.get_category_from_index(indices)

        if categories[0] != 0:
            print(f"Harmful content detected: index={indices.cpu().numpy()}, category={categories[0]}")

        return hidden_states, indices, categories


LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_START_DOCSTRING,
)
class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAVA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`LlavaProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The LLAVA model which consists of a vision backbone and a language model.""",
    LLAVA_START_DOCSTRING,
)


class LlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    """
    LLaVA model for conditional generation combining vision and language.

    This model integrates a vision encoder (UniTok) with a language model,
    allowing for multi-modal understanding and generation tasks.

    Args:
        config (`LlavaConfig`):
            Model configuration.
    """
    def __init__(self, config: LlavaConfig):
        super().__init__(config)


        unitok_cfg = Args()
        unitok = UniTok(unitok_cfg).cuda()
        self.vision_tower = unitok

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layer: int, vision_feature_select_strategy: str, bpda
    ):
        """
        Extract and project image features from the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`):
                Input images.
            vision_feature_layer (`int`):
                Index of the vision layer to extract features from.
            vision_feature_select_strategy (`str`):
                Strategy for selecting vision features ("default" or "full").

        Returns:
            image_features (`torch.FloatTensor`): Projected image features.
            indices (`torch.LongTensor`): Classification codebook indices.
            categories (`list`): Predicted category IDs.

        Raises:
            ValueError: If harmful content is detected during classification.
        """
        image_outputs = self.vision_tower.encoder_image_to_hidden(pixel_values) if not bpda else self.vision_tower.encoder_image_to_hidden_bpda(pixel_values)

        selected_image_feature = image_outputs
        image_features, indices, categories = self.multi_modal_projector(selected_image_feature)

        return image_features, indices, categories

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        """
        Merge text embeddings with image features at special image token positions.

        Args:
            image_features (`torch.FloatTensor`):
                Projected image features.
            inputs_embeds (`torch.FloatTensor`):
                Text token embeddings.
            input_ids (`torch.LongTensor`):
                Input token IDs.
            attention_mask (`torch.Tensor`):
                Attention mask for the input.
            labels (`torch.LongTensor`, *optional*):
                Labels for language modeling.

        Returns:
            final_embedding (`torch.FloatTensor`): Merged embeddings.
            final_attention_mask (`torch.Tensor`): Updated attention mask.
            final_labels (`torch.LongTensor`, *optional*): Updated labels.
            position_ids (`torch.LongTensor`): Position IDs for the merged sequence.
        """
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )

        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )
        final_embedding = final_embedding.to(image_features.dtype)
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        bpda = False,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            legacy_processing = (
                                    (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
                                ) or (input_ids.shape[-1] == 1 and pixel_values is not None)

        image_features = None
        if pixel_values is not None:
            image_features, indices, categories = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                bpda=bpda
            )

        if legacy_processing:
            logger.warning_once(
                "Expanding inputs for image tokens in LLaVa should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
            if input_ids.shape[1] != 1:
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            else:
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)[-target_length:]

        elif image_features is not None:
            n_image_tokens = (input_ids == self.config.image_token_index).sum(dim=-1)[0].item()
            n_image_features = image_features.shape[1]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (
                (input_ids == self.config.image_token_index)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        self.language_model = self.language_model.to(inputs_embeds.dtype)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs[0]

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation, handling caching and image inputs appropriately.

        Args:
            input_ids (`torch.LongTensor`):
                Input token IDs.
            past_key_values (*optional*):
                Cached key-value states from previous generation steps.
            inputs_embeds (*optional*):
                Input embeddings (alternative to input_ids).
            pixel_values (*optional*):
                Image pixel values.
            attention_mask (*optional*):
                Attention mask.
            cache_position (*optional*):
                Position indices for cached values.
            num_logits_to_keep (*optional*):
                Number of logits to compute.

        Returns:
            model_inputs (`dict`): Dictionary containing prepared inputs for the model.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


def normalize_01_into_pm1(x):
    """
    Normalize tensor values from [0, 1] range to [-1, 1] range.

    Args:
        x (`torch.Tensor`): Input tensor with values in [0, 1] range.

    Returns:
        `torch.Tensor`: Normalized tensor with values in [-1, 1] range.
    """
    return x * 2.0 - 1.0

unitok_preprocess = transforms.Compose([
    transforms.Resize(int(256 * 1.125)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize_01_into_pm1,
])

