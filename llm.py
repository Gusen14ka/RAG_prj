from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Llm:
    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-3B-Instruct",
            gpu_mode: bool = True
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if gpu_mode:
            self._load_model_gpu()
        else:
            self._load_model_cpu()

    def _load_model_cpu(self) -> None:
        # Задаём поле model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,   # float16 — вдвое меньше памяти чем float32
            device_map="CPU",
        )
        self.model.eval()

    def _load_model_gpu(self) -> None:
        device = torch.device("cuda")
        # Задаём поле model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,   # float16 — вдвое меньше памяти чем float32
        ).to(device) # type: ignore
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