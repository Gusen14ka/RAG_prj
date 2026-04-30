from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Llm:
    def __init__(
            self,
            model_name: str = "/app/models/qwen2.5-3b",
            gpu_mode: bool = True
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self._load_model(gpu_mode)

    def _load_model(self, gpu_mode: bool) -> None:
        # Задаём поле model
        if torch.cuda.is_available() and gpu_mode:
            print("GPU MODE acticated")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to("cuda") # type: ignore
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=torch.float16,
            )

        self.model.eval()

    def call_llm(self, query: str, system_promt: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_promt},
            {"role": "user", "content": query},
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — детерминированный результат
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Декодируем только новые токены (без промпта)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return result