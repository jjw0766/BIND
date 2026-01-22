import time
from grapheme import graphemes
import lightning as L
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

from src.model.utils import apply_neftune
from src.metrics.ChrF import chrf_corpus
from src.tokenizer.modeling_tokenizer import SentenceTokenizer

class LitInstructionModel(L.LightningModule):
    def __init__(
        self,
        base_model_name='Qwen/Qwen3-0.6B',
        use_qlora=True,
        lr=5e-5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        epochs=10,
        neftune_alpha=0,
        user_inference_sentence_tokenizer=False,
        inference_sentence_min_length=128,
        inference_sentence_max_length=64,
        inference_sentence_n_overlap=3,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.use_qlora = use_qlora
        self.lr = lr
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.use_inference_sentence_tokenizer = user_inference_sentence_tokenizer
        self.inference_sentence_n_overlap = inference_sentence_n_overlap

        if use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                quantization_config=quantization_config
            )
            model.train()
            if neftune_alpha:
                model = apply_neftune(model, neftune_alpha)   
                print('neftune applied') 
            model = prepare_model_for_kbit_training(model)
    
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
            )
            model.train()
            if neftune_alpha:
                model = apply_neftune(model, neftune_alpha)   
                print('neftune applied') 
        self.model = model

        if user_inference_sentence_tokenizer:
            self.sentence_tokenizer = SentenceTokenizer(
                min_length=inference_sentence_min_length,
                max_length=inference_sentence_max_length,
                n_overlap=inference_sentence_n_overlap,
                roll=False
            )
        else:
            self.sentence_tokenizer = None  
        

    def get_prompt(self, sentence_noisy, sentence=None, mode='train'):
        if mode=='train':
            messages = [
                {"role": "user", "content": f"Please decode the obfuscated text. Text: {sentence_noisy}"},
                {"role": "assistant", "content": sentence}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
            )
        elif mode=='inference':
            messages = [
                {"role": "user", "content": f"Please decode the obfuscated text. Text: {sentence_noisy}"},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
            )
        return prompt
    
    def batch_tokenize(self, batch, mode):
        prompts = []
        for sentence_noisy, sentence in zip(batch['sentence_noisy'], batch['sentence']):
            prompt = self.get_prompt(sentence_noisy, sentence, mode)
            prompts.append(prompt)
        inputs = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True, padding_side='left')
        return inputs

    def forward(self, batch):
        inputs = self.batch_tokenize(batch, mode='train')
        outputs = self.model(
            input_ids=inputs['input_ids'].to('cuda'),
            attention_mask=inputs['attention_mask'].to('cuda'),
            labels=inputs['input_ids'].to('cuda')
        )
        return outputs.loss, outputs.logits

    
    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.log('train_loss', loss, batch_size=len(batch['sentence_noisy']), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sentences_denoised, times = self.predict_step(batch, batch_idx)
        score = chrf_corpus(sentences_denoised, batch['sentence'])['f1']
        self.log('valid_score', score, batch_size=len(batch['sentence_noisy']))
        return score
    
    def predict_step_(self, batch, batch_idx):
        inputs = self.batch_tokenize(batch, mode='inference')
        outputs = self.model.generate(
            input_ids = inputs['input_ids'].to('cuda'),
            attention_mask = inputs['attention_mask'].to('cuda')
        )
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded
    
    def predict_step(self, batch, batch_idx):
        if not self.use_inference_sentence_tokenizer:
            times = []
            start = time.time()
            decoded = self.predict_step_(batch, batch_idx)
            end = time.time()
            times.append(end-start)
            return decoded, times
        else:
            sentences_noisy = batch['sentence_noisy']
            sentences_denoised = []
            times = []
            if self.inference_sentence_n_overlap > 1:
                for sentence_noisy in sentences_noisy:
                    start = time.time()
                    sentence_denoised_chunks_overlapped = []
                    sentence_noisy_chunks = self.sentence_tokenizer.split_text(sentence_noisy)
                    sentence_noisy_chunks_overlapped = self.sentence_tokenizer.overlap(sentence_noisy_chunks)
                    for start_idx, end_idx, sentence_noisy_chunk in sentence_noisy_chunks_overlapped:
                        mini_batch = {
                            'sentence_noisy': [sentence_noisy_chunk],
                            'sentence': [None]
                        }
                        sentence_denoised_chunk_untruncked = self.predict_step_(mini_batch, 0)[0]
                        sent_denoised_chunk_truncked = sentence_denoised_chunk_untruncked[:len(list(graphemes(sentence_noisy_chunk)))]
                        sentence_denoised_chunks_overlapped.append((start_idx, end_idx, sent_denoised_chunk_truncked))
                    sentence_denoised = self.sentence_tokenizer.decode_overlap(sentence_denoised_chunks_overlapped)
                    sentences_denoised.append(sentence_denoised)
                    end = time.time()
                    times.append(end-start)
            else:
                for sentence_noisy in sentences_noisy:
                    start = time.time()
                    sentence_denoised_chunks = []
                    sentence_noisy_chunks = self.sentence_tokenizer.split_text(sentence_noisy)
                    for sentence_noisy_chunk in sentence_noisy_chunks:
                        mini_batch = {
                            'sentence_noisy': [sentence_noisy_chunk],
                            'sentence': [None]
                        }
                        sentence_denoised_chunk_untruncked = self.predict_step_(mini_batch, 0)[0]
                        sent_denoised_chunk_truncked = sentence_denoised_chunk_untruncked[:len(list(graphemes(sentence_noisy_chunk)))]
                        sentence_denoised_chunks.append(sent_denoised_chunk_truncked)
                    sentence_denoised = ''.join(sentence_denoised_chunks)
                    sentences_denoised.append(sentence_denoised)
                    end = time.time()
                    times.append(end-start)
            return sentences_denoised, times
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,           # 또는 2e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,          # 총 epoch 수
            eta_min=1e-6        # 최소 learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 매 epoch마다 갱신
                "frequency": 1,
            }
        }