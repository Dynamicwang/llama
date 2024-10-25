# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """

        参数：
            ckpt_dir (str): 模型文件路径
            tokenizer_path (str): 分词器文件路径
            max_seq_len (int): 输入序列最大长度
            max_batch_size (int): 批量大小
            model_parallel_size (int): 模型并行大小
            seed (int): 随机种子


        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        # 如果没有初始化分布式进程组，则初始化后端为nccl
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        # 
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        # 通过环境变量获取本地进程的rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 设置当前进程的设备使用第几块GPU
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        # 获取模型的路径
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        #检查模型权重文件存在
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        # 检查模型权重文件数量是否等于模型并行大小
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        
        # 根据模型并行的rank获取当前进程对应的模型权重文件
        ckpt_path = checkpoints[get_model_parallel_rank()]

        # 加载模型权重到内存
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # 获取模型相关的参数信息
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # 根据传入的参数和从配置文件读取的参数，初始化ModelArgs对象
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        # 初始化tokenizer
        tokenizer = Tokenizer(model_path=tokenizer_path)

        #通过分词器模型获取词表大小
        model_args.vocab_size = tokenizer.n_words

        # 设置权重参数的精度为半精度
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        # 根据模型参数，定义模型结构
        model = Transformer(model_args)

        # 加载模型权重到模型中
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        # 将初始化好的模型和tokenizer封装到Llama对象中
        return Llama(model, tokenizer)


    # build返回的就是调用该构造函数返回的Llama对象
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        参数：
            prompt_tokens: 所有对话的token列表
            max_gen_len: 生成文本的最大长度
            temperature:  改变生成文本的随机性， 通过改变生成结果的分布来影响结果
            top_p: 
            logprobs: 是否返回每个token的对数概率
            echo: 是否在生成文本中包含原始的prompt

        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        
        # 得到输入数据的batch size
        bsz = len(prompt_tokens)

        # 检查batch size是否超过模型的最大batch size
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # 得到不同batch的prompt的最大和最小长度
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        # 检查prompt的长度是否超过了模型的最大输入序列长度
        assert max_prompt_len <= params.max_seq_len

        # 在最大输入序列长度、最大生成长度+最大输入prompt 之间取最小值作为实际的生成长度
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # 初始化一个全为pad_id的tokens张量，大小为batch size * total_len
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")

        # 将prompt的token填充到tokens张量中， 在每个batch中， 前面存放输入prompt的token, 后面临时使用pad填充
        # 例如：prompt_tokens = [[1, 2, 3], [4, 5, 6, 7]] tokens = [[1, 2, 3, pad, pad, pad], [4, 5, 6, 7, pad, pad]]
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        
        # 如果要得到每个token的概率，则初始化一个全为0的token_logprobs张量，大小为batch size * total_len
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        #    
        prev_pos = 0

        # 初始化一个全为False的eos_reached张量，大小为batch size， 标记某个batch中的prompt是否已经生成完毕
        eos_reached = torch.tensor([False] * bsz, device="cuda")

        # 生成一个mask, 大小为batch size * total_len, 用来标记哪些位置是prompt的token, 哪些位置是pad， pad的位置为False, prompt的位置为True
        # 如：tokens = [[1, 2, 3, pad, pad, pad], [4, 5, 6, 7, pad, pad]] input_text_mask = [[True, True, True, False, False, False], [True, True, True, True, False, False]]
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        # 从最小的输入prompt的长度开始， 逐步增加prompt的长度，直到达到最大长度 
        for cur_pos in range(min_prompt_len, total_len):
            # 所有batch的prev_pos:cur_pos之间的token，作为模型的输入token， 模型返回的结果为[batchsize,size ,vocab_size]
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # 如果temperature大于0，则使用top_p采样，否则使用argmax采样
            if temperature > 0:
                # logits[:, -1]表示最后一个token的logits， 除以temperature后，再使用softmax函数，得到每个token的概率分布
                # temperature越大，则概率分布越平滑，模型生成的文本等价多样化和随机。
                # temperature越小，则概率分布越尖锐， 模型生成的文本更加确定和集中。
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # 采样得到下一个token
                next_token = sample_top_p(probs, top_p)
            else:
                # 如果temperature小于等于0，则使用argmax采样，即选择概率最大的token
                next_token = torch.argmax(logits[:, -1], dim=-1)

            #上一步采样得到的next_token的维度为[batchsize, 1], 需要将其reshape为[batchsize], 然后将其添加到tokens中
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            # input_text_mask为使用pad的mask矩阵，如果当前位置为pad，则input_text_mask[:, cur_pos]为fasle， :表示取所有样本的该位置
            # torch.where(condition, x, y)函数表示如果condition为true，则返回x，否则返回y
            # 所以下面的意思是，根据cur_pos是否为pad，来决定是否将next_token添加到tokens中
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            # 将next_token添加到tokens中
            tokens[:, cur_pos] = next_token

            # 如果需要计算logprobs，则计算当前token的logprob
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            #
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    # 调用Llama对象的chat_completion方法，生成对话
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        参数：
            dialogs: List[Dialog] 对话列表，每个对话是一个消息列表
            temperature: 
            top_p: 
            max_gen_len: Optional[int] 最大生成长度
            logprobs: bool 是否返回每个生成的token的对数概率
        
        
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # 如果没有设置最大生成长度，则设置为模型的最大输入序列长度减1
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        
        # 
        prompt_tokens = []
        unsafe_requests = []

        # 遍历对话列表,执行完后，prompt_tokens中存储的是所有对话的prompt，如：[[dialog1 tokens], [dialog2 tokens], ...]
        # unsafe_requests中存储的是所有对话是否包含非法字符，如果有，则记录为true，如：[False, True, ...]
        for dialog in dialogs:
            # 所有的对话是否包含非法字符，如果有，则记录为true
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )

            # 检测对话的第一条消息是否是系统消息，如果是，则使用B_SYS作为前缀，E_SYS作为后缀,并将系统提示拼接到下一条消息之前。后面的保持不变
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            # 判断所有的对话顺序为，user, assistant, user, assistant, ...
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            # 将一组对话中的所有消息根据user和assistant进行拼接，并使用B_INST和E_INST作为user的prompt的前缀和后缀
            # 将拼接的对话转换为token，tokenizer.encode()函数将将文本转换为一个列表，列表中的每个元素是一个token的id
            # 最终得到的就是多轮对话的token列表，例如：[[user+assistant token], [user+assistant token], ...]
            # 使用sum()函数将列表中的所有元素进行拼接，得到的就是多轮对话的token列表, 最终的结果是[user+assistant token, user+assistant token, ...]
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            # 判断最后一个对话是用户, 否则报错, 因为模型只能接受用户输入
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            # 上面成组的将对话转换为token， 根据user和assistant成对， 那势必会剩下当前轮的user， 所以需要单独处理
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            # 将一组完成对话的tokens添加到prompt_tokens中
            prompt_tokens.append(dialog_tokens)

        # 根据上面生成的token, 进行对话结果的生成
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        # 将生成的token转换为文本， 分为两种情况， 如果logprobs为True， 则返回文本和logprobs， 否则只返回文本
        if logprobs:
            # 生成文本和logprobs, 根据unsafe参数决定是否返回原始文本, 如果unsafe为True, 则返回原始文本, 否则返回解码后的文本
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        # 如果logprobs为False， 则只返回文本，根据unsafe判断是否返回错误信息, 如果unsafe为True， 则返回错误信息 UNSAFE_ERROR
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # 对所有的概率进行降序排列
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算累积概率， 如果有一维数组 [a, b, c, d]，那么它的累积和结果会是 [a, a+b, a+b+c, a+b+c+d]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 得到一个mask ，如果累积概率大于p，则mask为True，否则为False
    # 例如：[0.1, 0.2, 0.3, 0.4] p=0.3，则mask为[True, True, False, False]
    mask = probs_sum - probs_sort > p
    # 将mask为True的值设置为0.0, 
    probs_sort[mask] = 0.0
    # 对mask以后得概率进行归一化
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 对归一化后的概率进行采样,采样得到的是对应概率所在位置的索引
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # 根据索引从probs_idx中获取对应的token
    # 例如：probs_idx为原始概率排序后的索引表，例如原始【2,3,1,4],经过排序后为【4， 3， 2， 1】，得然后probs_idx为【3， 1， 0， 2】
    #  假如next_token为[0], 则probs_idx[next_token]为[3]，即原始概率中第3个位置的概率
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


