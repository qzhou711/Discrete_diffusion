import pickle
import os
from pathlib import Path
from typing import List, Tuple

from accelerate import init_empty_weights
import torch

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration_ck_action_head
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.pretrain import PretrainSolverBase_ck_action_head

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class ItemProcessor(ItemProcessorBase):
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(PretrainSolverBase_ck_action_head):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--time_horizon", type=int, default=5)
        parser.add_argument("--preprocess", default='true', choices=['true', 'false'])
        parser.add_argument("--with_state", action='store_true')
        parser.add_argument("--with_wrist", action='store_true')
        parser.add_argument("--with_action", action='store_true')
        parser.add_argument("--with_world_model", action='store_true')
        parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
        parser.add_argument("--tokenizer_path", type=str, default="../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af")
        # LoRA parameters
        parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
        parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated list of target modules for LoRA")
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck_action_head, None):

        # Check if init_from is a valid checkpoint (has config.json)
        # If not, it might be a checkpoint directory without config, use the base model path
        init_path = Path(init_from)
        has_config = (init_path / "config.json").exists() if init_path.is_dir() else False
        
        # If it's a checkpoint directory without config.json, use the base model from args.init_from
        # This handles the case where auto_resume finds a checkpoint but it's incomplete
        if not has_config and hasattr(self.args, 'init_from') and self.args.init_from:
            # Check if the original init_from is different and valid
            original_init = Path(self.args.init_from)
            if original_init.exists() and ((original_init / "config.json").exists() if original_init.is_dir() else True):
                # Use original init_from for config, but we'll load weights from resume_path later
                base_model_path = str(original_init)
            else:
                base_model_path = init_from
        else:
            base_model_path = init_from

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        if self.dp_rank == 0:
            # Disable automatic adapter loading to avoid PEFT version issues
            # Temporarily rename adapter_config.json if it exists to prevent auto-loading
            adapter_config_path = Path(base_model_path) / "adapter_config.json" if Path(base_model_path).is_dir() else None
            adapter_renamed = False
            temp_adapter_config = None
            if adapter_config_path and adapter_config_path.exists():
                temp_adapter_config = str(adapter_config_path) + ".tmp"
                try:
                    os.rename(str(adapter_config_path), temp_adapter_config)
                    adapter_renamed = True
                except Exception:
                    adapter_renamed = False
            
            try:
                model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                    base_model_path,
                    action_dim=self.args.action_dim,
                    time_horizon=self.args.time_horizon,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                )
            finally:
                # Restore adapter config if it was renamed
                if adapter_renamed and temp_adapter_config:
                    try:
                        os.rename(temp_adapter_config, str(adapter_config_path))
                    except Exception:
                        pass
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    base_model_path,
                    action_dim=self.args.action_dim,
                    time_horizon=self.args.time_horizon,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGeneration_ck_action_head(config)

        del model.model.vqmodel

        # Apply LoRA if requested
        # Note: LoRA should be applied on all ranks to ensure model structure consistency for FSDP
        if hasattr(self.args, 'use_lora') and self.args.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library is required for LoRA training. Please install it with: pip install peft")
            
            # Check if model already has LoRA adapters (from checkpoint)
            # If so, we don't need to apply again
            if hasattr(model, 'peft_config') and len(model.peft_config) > 0:
                if self.dp_rank == 0:
                    print("Model already has LoRA adapters loaded from checkpoint, skipping LoRA application")
            else:
                target_modules = [m.strip() for m in self.args.lora_target_modules.split(',')]
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                )
                model = get_peft_model(model, lora_config)
                if self.dp_rank == 0:
                    model.print_trainable_parameters()

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run_with_eval_awm_w()