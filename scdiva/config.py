# scdiva/config.py
import json

class Config:
    # 数据来源目录
    DATA_ROOT_DIR = "/data/wangmingxuan/merged_10_files"
    OUTPUT_DIR = "scdiva_output_pretrain_from_scratch"

    # 维度配置
    SCGPT_DIM = 512
    HIDDEN_DIM = 512
    N_LAYERS = 12
    N_HEADS = 8
    DROPOUT = 0.1

    # FFN hidden size：保持你原 pretrain = 4*hidden
    D_HID = 4 * HIDDEN_DIM

    # 词表配置（= token id 空间）
    VOCAB_SIZE = 41818
    PAD_TOKEN_ID = 0
    MASK_TOKEN_ID = 1

    # 特殊 token id（你已确认）
    PAD_FILL_GENE_ID = 41815  # value = -2, padding to 1200
    BOS_GENE_ID = 41816       # value = 0, sequence start
    EOS_GENE_ID = 41817       # value = -3, sequence end

    # 训练配置
    MAX_GENE_LEN = 1200
    BATCH_SIZE = 24
    GRAD_ACCUM_STEPS = 8
    
    # RoPE 配置
    ROPE_THETA = 10000.0
    ROPE_MAX_LEN = MAX_GENE_LEN * 2 + 1  # + latent token

    # Streaming shuffle buffer
    SHUFFLE_BUFFER = 50000

    # Eval take
    EVAL_TAKE = 20000

    # 伪装成 HF config
    def to_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith("__") and not callable(v)}

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)
