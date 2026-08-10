/* Course index. Plain global - no fetch(), so this works from file:// */
window.COURSE = {
 "title": "Gen AI & Agentic AI",
 "subtitle": "From what a gradient is, to shipping your own agent.",
 "phases": [
  {
   "n": 1,
   "name": "Foundations",
   "tag": "The ideas everything else stands on"
  },
  {
   "n": 2,
   "name": "Inside the machine",
   "tag": "What a language model actually is"
  },
  {
   "n": 3,
   "name": "Building with LLMs",
   "tag": "Turning a model into something useful"
  },
  {
   "n": 4,
   "name": "Agentic AI",
   "tag": "Systems that decide and act"
  },
  {
   "n": 5,
   "name": "Production",
   "tag": "Making it real, safe and affordable"
  },
  {
   "n": 6,
   "name": "Capstone",
   "tag": "Prove it end to end"
  },
  {
   "n": 7,
   "name": "Build track",
   "tag": "Make your own model. Runs alongside the main course."
  }
 ],
 "modules": [
  {
   "n": 1,
   "id": "m01",
   "file": "m01-ml-refresher.html",
   "phase": 1,
   "title": "What machine learning you still need",
   "promise": "Refresh only the machine-learning ideas that still matter for building with AI, then set up the Python workspace you will use for the next seventeen modules.",
   "hours": 3,
   "concepts": [
    "supervised vs unsupervised",
    "features and labels",
    "train/test split",
    "overfitting",
    "loss function",
    "gradient descent",
    "why classic ML still matters",
    "virtual environments"
   ],
   "videos": [
    {
     "id": "i_LwzRVP7bg",
     "t": "Machine Learning for Everybody – Full Course",
     "c": "freeCodeCamp.org",
     "v": 10253468,
     "m": 233,
     "d": "2022-09",
     "core": true
    },
    {
     "id": "BUTjcAjfMgY",
     "t": "ML Foundations for AI Engineers (in 34 Minutes)",
     "c": "Shaw Talebi",
     "v": 446080,
     "m": 34,
     "d": "2025-05",
     "core": true
    },
    {
     "id": "AMdG7IjgSPM",
     "t": "Python Tutorial: UV - A Faster, All-in-One Package Manager to Replace Pip and Venv",
     "c": "Corey Schafer",
     "v": 273372,
     "m": 27,
     "d": "2025-04",
     "core": false
    },
    {
     "id": "U07MHi4Suj8",
     "t": "How to Actually Learn LLMs in 2026 | Ex-Google, Microsoft Engineer",
     "c": "Aishwarya Srinivasan",
     "v": 124416,
     "m": 13,
     "d": "2026-02",
     "core": false
    }
   ],
   "lab": {
    "t": "Set up your AI workspace",
    "g": "Create a virtual environment, install your first packages, and call a model two different ways."
   },
   "mini": {
    "t": "Two providers, one script",
    "g": "Make a single script answer the same question using a hosted model and a local model, switching by config only."
   },
   "widgets": [
    "predictor"
   ]
  },
  {
   "n": 2,
   "id": "m02",
   "file": "m02-neural-networks.html",
   "phase": 1,
   "title": "How a neural network learns",
   "promise": "Understand what a neural network really is and how it teaches itself, so that nothing later in the course has to feel like magic.",
   "hours": 4,
   "concepts": [
    "neuron",
    "weights and biases",
    "activation function",
    "forward pass",
    "backpropagation",
    "learning rate",
    "epochs and batches",
    "why depth helps"
   ],
   "videos": [
    {
     "id": "aircAruvnKk",
     "t": "But what is a neural network? | Deep learning chapter 1",
     "c": "3Blue1Brown",
     "v": 23833921,
     "m": 18,
     "d": "2017-10",
     "core": true
    },
    {
     "id": "IHZwWFHWa-w",
     "t": "Gradient descent, how neural networks learn | Deep Learning Chapter 2",
     "c": "3Blue1Brown",
     "v": 9419092,
     "m": 20,
     "d": "2017-10",
     "core": true
    },
    {
     "id": "Ilg3gGewQ5U",
     "t": "Backpropagation, intuitively | Deep Learning Chapter 3",
     "c": "3Blue1Brown",
     "v": 6154992,
     "m": 12,
     "d": "2017-11",
     "core": true
    },
    {
     "id": "tIeHLnjs5U8",
     "t": "Backpropagation calculus | Deep Learning Chapter 4",
     "c": "3Blue1Brown",
     "v": 3871982,
     "m": 10,
     "d": "2017-11",
     "core": false
    },
    {
     "id": "VMj-3S1tku0",
     "t": "The spelled-out intro to neural networks and backpropagation: building micrograd",
     "c": "Andrej Karpathy",
     "v": 3878022,
     "m": 145,
     "d": "2022-08",
     "core": false
    }
   ],
   "lab": {
    "t": "Train a network by hand",
    "g": "Build a tiny neural network with no deep-learning library and watch its loss fall."
   },
   "mini": {
    "t": "Teach it a new shape",
    "g": "Train your network to separate a pattern it has never seen, and plot the decision boundary."
   },
   "widgets": [
    "neuron"
   ]
  },
  {
   "n": 3,
   "id": "m03",
   "file": "m03-tokens-embeddings.html",
   "phase": 1,
   "title": "How text becomes numbers",
   "promise": "See exactly how a sentence turns into tokens, and how tokens turn into numbers a computer can do maths on.",
   "hours": 4,
   "concepts": [
    "tokenization",
    "byte pair encoding",
    "vocabulary",
    "token IDs",
    "embeddings",
    "cosine similarity",
    "vector space",
    "context window"
   ],
   "videos": [
    {
     "id": "zduSFxRajkE",
     "t": "Let's build the GPT Tokenizer",
     "c": "Andrej Karpathy",
     "v": 1161030,
     "m": 133,
     "d": "2024-02",
     "core": true
    },
    {
     "id": "gl1r1XV0SLw",
     "t": "What is a Vector Database? Powering Semantic Search & AI Applications",
     "c": "IBM Technology",
     "v": 909567,
     "m": 9,
     "d": "2025-03",
     "core": true
    },
    {
     "id": "viZrOnJclY0",
     "t": "Word Embedding and Word2Vec, Clearly Explained!!!",
     "c": "StatQuest with Josh Starmer",
     "v": 584306,
     "m": 16,
     "d": "2023-03",
     "core": true
    },
    {
     "id": "nKSk_TiR8YA",
     "t": "Most devs don't understand how LLM tokens work",
     "c": "Matt Pocock",
     "v": 330146,
     "m": 10,
     "d": "2025-09",
     "core": false
    },
    {
     "id": "wgfSDrqYMJ4",
     "t": "What are Word Embeddings?",
     "c": "IBM Technology",
     "v": 260433,
     "m": 8,
     "d": "2024-08",
     "core": false
    }
   ],
   "lab": {
    "t": "Look inside the tokenizer",
    "g": "Tokenize your own text, embed it locally, and measure which sentences are closest in meaning."
   },
   "mini": {
    "t": "Find the odd one out",
    "g": "Given five sentences, have your program pick the one that does not belong - using only embeddings."
   },
   "widgets": [
    "tokenizer",
    "embeddings"
   ]
  },
  {
   "n": 4,
   "id": "m04",
   "file": "m04-transformers.html",
   "phase": 2,
   "title": "The transformer, and what attention means",
   "promise": "Open up the architecture behind every modern language model and understand the single idea that makes it work.",
   "hours": 5,
   "concepts": [
    "attention",
    "query key value",
    "multi-head attention",
    "positional encoding",
    "residual connections",
    "feed-forward layers",
    "decoder-only models",
    "why transformers scaled"
   ],
   "videos": [
    {
     "id": "wjZofJX0v4M",
     "t": "Transformers, the tech behind LLMs | Deep Learning Chapter 5",
     "c": "3Blue1Brown",
     "v": 10837072,
     "m": 27,
     "d": "2024-04",
     "core": true
    },
    {
     "id": "eMlx5fFNoYc",
     "t": "Attention in transformers, step-by-step | Deep Learning Chapter 6",
     "c": "3Blue1Brown",
     "v": 4369296,
     "m": 26,
     "d": "2024-04",
     "core": true
    },
    {
     "id": "9-Jl0dxWQs8",
     "t": "How might LLMs store facts | Deep Learning Chapter 7",
     "c": "3Blue1Brown",
     "v": 2173236,
     "m": 22,
     "d": "2024-08",
     "core": true
    },
    {
     "id": "zxQyTK8quyY",
     "t": "Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!",
     "c": "StatQuest with Josh Starmer",
     "v": 1161820,
     "m": 36,
     "d": "2023-07",
     "core": false
    },
    {
     "id": "kCc8FmEb1nY",
     "t": "Let's build GPT: from scratch, in code, spelled out.",
     "c": "Andrej Karpathy",
     "v": 7609153,
     "m": 116,
     "d": "2023-01",
     "core": false
    }
   ],
   "lab": {
    "t": "Watch attention happen",
    "g": "Pull real attention weights out of a small open model and see which words look at which."
   },
   "mini": {
    "t": "Attention detective",
    "g": "Find a sentence where attention clearly resolves an ambiguous word, and explain what you found."
   },
   "widgets": [
    "attention"
   ]
  },
  {
   "n": 5,
   "id": "m05",
   "file": "m05-llm-training.html",
   "phase": 2,
   "title": "How an LLM is trained",
   "promise": "Follow a model from raw internet text to a polished assistant that can reason, and learn why each training stage has to exist.",
   "hours": 5,
   "concepts": [
    "pretraining",
    "next-token prediction",
    "supervised fine-tuning",
    "RLHF",
    "RLVR",
    "reasoning models",
    "test-time compute",
    "chain of thought"
   ],
   "videos": [
    {
     "id": "7xTGNNLPyMI",
     "t": "Deep Dive into LLMs like ChatGPT",
     "c": "Andrej Karpathy",
     "v": 8386781,
     "m": 211,
     "d": "2025-02",
     "core": true
    },
    {
     "id": "gY4Z-9QlZ64",
     "t": "DeepSeek is a Game Changer for AI - Computerphile",
     "c": "Computerphile",
     "v": 1552215,
     "m": 19,
     "d": "2025-01",
     "core": true
    },
    {
     "id": "hhiLw5Q_UFg",
     "t": "John Schulman - Reinforcement Learning from Human Feedback: Progress and Challenges",
     "c": "UC Berkeley EECS",
     "v": 87248,
     "m": 63,
     "d": "2023-04",
     "core": true
    },
    {
     "id": "qPN_XZcJf_s",
     "t": "Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!",
     "c": "StatQuest with Josh Starmer",
     "v": 64313,
     "m": 18,
     "d": "2025-05",
     "core": false
    },
    {
     "id": "K5WPr5dtne0",
     "t": "State of LLMs 2026: RLVR, GRPO, Inference Scaling — Sebastian Raschka",
     "c": "The MAD Podcast with Matt Turck",
     "v": 22388,
     "m": 68,
     "d": "2026-01",
     "core": false
    }
   ],
   "lab": {
    "t": "Base model vs reasoning model",
    "g": "Run the same hard question through two model types and compare answers, tokens and cost."
   },
   "mini": {
    "t": "Where reasoning pays",
    "g": "Find one question where a reasoning model wins clearly and one where it wastes money."
   },
   "widgets": []
  },
  {
   "n": 6,
   "id": "m06",
   "file": "m06-model-landscape.html",
   "phase": 2,
   "title": "Picking a model, and what it costs",
   "promise": "Choose the right model for a job in 2026, and predict the bill before you get it.",
   "hours": 4,
   "concepts": [
    "model families",
    "context window size",
    "input vs output pricing",
    "prompt caching",
    "KV cache hit rate",
    "model routing",
    "reasoning effort",
    "benchmarks and saturation"
   ],
   "videos": [
    {
     "id": "-0HRzXk8vlk",
     "t": "Why AI Tokens are so Expensive - Computerphile",
     "c": "Computerphile",
     "v": 702677,
     "m": 25,
     "d": "2026-07",
     "core": true
    },
    {
     "id": "pYax2rupKEY",
     "t": "How to Choose Large Language Models: A Developer’s Guide to LLMs",
     "c": "IBM Technology",
     "v": 111228,
     "m": 6,
     "d": "2025-05",
     "core": true
    },
    {
     "id": "u57EnkQaUTY",
     "t": "What is Prompt Caching? Optimize LLM Latency with AI Transformers",
     "c": "IBM Technology",
     "v": 99696,
     "m": 9,
     "d": "2026-02",
     "core": true
    },
    {
     "id": "AVQzG2MY858",
     "t": "LLM vs. SLM vs. FM: Choosing the Right AI Model",
     "c": "IBM Technology",
     "v": 77572,
     "m": 14,
     "d": "2026-01",
     "core": false
    },
    {
     "id": "fjd2hm6-qtM",
     "t": "What is OpenRouter | All about OpenRouter in 10 minutes",
     "c": "codebasics",
     "v": 35563,
     "m": 10,
     "d": "2026-03",
     "core": false
    }
   ],
   "lab": {
    "t": "Build a cost model",
    "g": "Measure real token usage for a task and calculate what a thousand runs would cost on five different models."
   },
   "mini": {
    "t": "A router that saves money",
    "g": "Route easy questions to a cheap model and hard ones to a strong one, and report the saving."
   },
   "widgets": [
    "cost",
    "cacheprefix"
   ]
  },
  {
   "n": 7,
   "id": "m07",
   "file": "m07-prompting.html",
   "phase": 3,
   "title": "Prompting that actually works",
   "promise": "Learn the handful of prompt patterns that reliably change output quality, and stop guessing.",
   "hours": 3,
   "concepts": [
    "zero-shot and few-shot",
    "system prompts",
    "chain of thought prompting",
    "output format control",
    "delimiters",
    "prompt iteration",
    "temperature",
    "common prompt mistakes"
   ],
   "videos": [
    {
     "id": "_ZvnD73m40o",
     "t": "Prompt Engineering Tutorial – Master ChatGPT and LLM Responses",
     "c": "freeCodeCamp.org",
     "v": 2744565,
     "m": 41,
     "d": "2023-09",
     "core": true
    },
    {
     "id": "EWvNQjAaOHw",
     "t": "How I use LLMs",
     "c": "Andrej Karpathy",
     "v": 2616829,
     "m": 131,
     "d": "2025-02",
     "core": true
    },
    {
     "id": "ysPbXH0LpIE",
     "t": "Prompting 101 | Code w/ Claude",
     "c": "Anthropic",
     "v": 473270,
     "m": 24,
     "d": "2025-07",
     "core": true
    }
   ],
   "lab": {
    "t": "Run a prompt experiment",
    "g": "Test five prompt variants on the same task and score them automatically instead of by feel."
   },
   "mini": {
    "t": "Beat the baseline prompt",
    "g": "Improve a weak prompt until it scores at least 30% higher on your own test set."
   },
   "widgets": []
  },
  {
   "n": 8,
   "id": "m08",
   "file": "m08-context-engineering.html",
   "phase": 3,
   "title": "Context engineering",
   "promise": "Manage the model's limited working memory on purpose. This is the skill that separates a demo from a system.",
   "hours": 5,
   "concepts": [
    "context engineering",
    "the six context layers",
    "context rot",
    "context poisoning",
    "distraction",
    "confusion",
    "clash",
    "compaction",
    "memory"
   ],
   "videos": [
    {
     "id": "-uW5-TaVXu4",
     "t": "Most devs don’t understand how context windows work",
     "c": "Matt Pocock",
     "v": 258496,
     "m": 9,
     "d": "2025-10",
     "core": true
    },
    {
     "id": "jLuwLJBQkIs",
     "t": "Context Engineering Clearly Explained",
     "c": "Tina Huang",
     "v": 208214,
     "m": 12,
     "d": "2025-08",
     "core": true
    },
    {
     "id": "TUjQuC4ugak",
     "t": "Context Rot: How Increasing Input Tokens Impacts LLM Performance",
     "c": "Chroma",
     "v": 185896,
     "m": 7,
     "d": "2025-07",
     "core": true
    },
    {
     "id": "vtugjs2chdA",
     "t": "Context Engineering Our Way to Long-Horizon Agents: LangChain’s Harrison Chase",
     "c": "Sequoia Capital",
     "v": 115484,
     "m": 39,
     "d": "2026-01",
     "core": false
    },
    {
     "id": "6_BcCthVvb8",
     "t": "Context Engineering for AI Agents with LangChain and Manus",
     "c": "LangChain",
     "v": 68174,
     "m": 60,
     "d": "2025-10",
     "core": false
    },
    {
     "id": "_IlTcWciEC4",
     "t": "Context Engineering for Agents - Lance Martin, LangChain",
     "c": "Latent Space",
     "v": 42908,
     "m": 63,
     "d": "2025-09",
     "core": false
    }
   ],
   "lab": {
    "t": "Pack a context window",
    "g": "Fill a fixed token budget from six competing sources and report exactly what got dropped."
   },
   "mini": {
    "t": "The honest context packer",
    "g": "Make your packer explain, in plain English, why each dropped item was dropped."
   },
   "widgets": [
    "context"
   ]
  },
  {
   "n": 9,
   "id": "m09",
   "file": "m09-tools-structured-output.html",
   "phase": 3,
   "title": "Getting exact data back, and calling your code",
   "promise": "Make a model return structured data you can rely on, and let it call your own Python functions.",
   "hours": 5,
   "concepts": [
    "JSON schema",
    "structured output",
    "function calling",
    "tool definitions",
    "Pydantic",
    "validation and retries",
    "constrained decoding",
    "tool choice"
   ],
   "videos": [
    {
     "id": "y5EmRr1O1h4",
     "t": "Structured Output in LangChain | Generative AI using LangChain | Video 5 | CampusX",
     "c": "CampusX",
     "v": 219658,
     "m": 68,
     "d": "2025-03",
     "core": true
    },
    {
     "id": "kE4BkATIl9c",
     "t": "OpenAI DevDay 2024 | Structured outputs for reliable applications",
     "c": "OpenAI",
     "v": 96532,
     "m": 40,
     "d": "2024-12",
     "core": true
    },
    {
     "id": "EzYaFF7ahKw",
     "t": "Tool Calling in LangChain | Generative AI using LangChain | Video 17 | CampusX",
     "c": "CampusX",
     "v": 95243,
     "m": 58,
     "d": "2025-04",
     "core": true
    },
    {
     "id": "3xUW1Do9zOs",
     "t": "Instructor and Pydantic - Structured LLM outputs for easy data extraction!",
     "c": "BugBytes",
     "v": 4885,
     "m": 16,
     "d": "2025-08",
     "core": false
    }
   ],
   "lab": {
    "t": "Make the model call your function",
    "g": "Define a tool, let the model choose it, run it, and feed the result back - all by hand."
   },
   "mini": {
    "t": "Three tools, no hints",
    "g": "Give the model three tools and make it pick correctly on five of five questions without being told which."
   },
   "widgets": [
    "toolcall"
   ]
  },
  {
   "n": 10,
   "id": "m10",
   "file": "m10-rag.html",
   "phase": 3,
   "title": "RAG - giving a model your own documents",
   "promise": "Build something that answers questions from your own files, and learn the three RAG designs and when each one fits.",
   "hours": 6,
   "concepts": [
    "chunking",
    "embeddings index",
    "vector search",
    "BM25 keyword search",
    "hybrid search",
    "reciprocal rank fusion",
    "reranking",
    "agentic RAG",
    "GraphRAG",
    "temporal drift"
   ],
   "videos": [
    {
     "id": "T-D1OfcDW1M",
     "t": "What is Retrieval-Augmented Generation (RAG)?",
     "c": "IBM Technology",
     "v": 1918635,
     "m": 6,
     "d": "2023-08",
     "core": true
    },
    {
     "id": "sVcwVQRHIc8",
     "t": "Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer",
     "c": "freeCodeCamp.org",
     "v": 1499794,
     "m": 153,
     "d": "2024-04",
     "core": true
    },
    {
     "id": "XvKiTfd6Xvo",
     "t": "The Complete Guide to Hybrid Search in RAG (BM25 + Embeddings + Reranker)",
     "c": "Dave Ebbelaar",
     "v": 21197,
     "m": 59,
     "d": "2026-05",
     "core": true
    },
    {
     "id": "nZnwExutgsY",
     "t": "GraphRAG in Python: Agentic AI with Knowledge Graphs",
     "c": "NeuralNine",
     "v": 7036,
     "m": 25,
     "d": "2026-07",
     "core": false
    }
   ],
   "lab": {
    "t": "Build a RAG system over your files",
    "g": "Index your own documents with hybrid search and reranking, then ask questions about them."
   },
   "mini": {
    "t": "Answer with receipts",
    "g": "Make your RAG answer cite the exact source chunk, and refuse to answer when it has no support."
   },
   "widgets": [
    "chunking"
   ]
  },
  {
   "n": 11,
   "id": "m11",
   "file": "m11-multimodal-finetuning.html",
   "phase": 3,
   "title": "Images, audio, and teaching a model your style",
   "promise": "Work with models that see and hear, then fine-tune a small model on your own data for a few cents.",
   "hours": 5,
   "concepts": [
    "multimodal models",
    "vision language models",
    "image generation",
    "diffusion",
    "fine-tuning",
    "LoRA",
    "QLoRA",
    "when not to fine-tune"
   ],
   "videos": [
    {
     "id": "1CIpzeNxIhU",
     "t": "How AI Image Generators Work (Stable Diffusion / Dall-E) - Computerphile",
     "c": "Computerphile",
     "v": 1062164,
     "m": 17,
     "d": "2022-10",
     "core": true
    },
    {
     "id": "pTaSDVz0gok",
     "t": "EASIEST Way to Fine-Tune a LLM and Use It With Ollama",
     "c": "Tech With Tim",
     "v": 345342,
     "m": 22,
     "d": "2025-06",
     "core": true
    },
    {
     "id": "t1caDsMzWBk",
     "t": "LoRA & QLoRA Fine-tuning Explained In-Depth",
     "c": "Mark Hennings",
     "v": 181296,
     "m": 14,
     "d": "2023-12",
     "core": true
    },
    {
     "id": "lOD_EE96jhM",
     "t": "What Are Vision Language Models? How AI Sees & Understands Images",
     "c": "IBM Technology",
     "v": 129247,
     "m": 9,
     "d": "2025-05",
     "core": false
    },
    {
     "id": "CcrC5zSv1iA",
     "t": "LLM Fine-Tuning Course – From Supervised FT to RLHF, LoRA, and Multimodal",
     "c": "freeCodeCamp.org",
     "v": 85990,
     "m": 716,
     "d": "2026-03",
     "core": false
    }
   ],
   "lab": {
    "t": "Fine-tune a small model",
    "g": "Train a LoRA adapter on a free GPU and compare the model before and after."
   },
   "mini": {
    "t": "Your own voice",
    "g": "Fine-tune on twenty examples of your own writing and get a noticeably different answer style."
   },
   "widgets": []
  },
  {
   "n": 12,
   "id": "m12",
   "file": "m12-agent-foundations.html",
   "phase": 4,
   "title": "Workflows, agents, and the loop",
   "promise": "Write an agent loop yourself with no framework, so you understand every moving part before anything hides it from you.",
   "hours": 5,
   "concepts": [
    "workflow vs agent",
    "the agent loop",
    "ReAct",
    "thought action observation",
    "planning",
    "tool selection",
    "human in the loop",
    "stopping conditions"
   ],
   "videos": [
    {
     "id": "F8NKVhkZZWI",
     "t": "What are AI Agents?",
     "c": "IBM Technology",
     "v": 1746419,
     "m": 12,
     "d": "2024-07",
     "core": true
    },
    {
     "id": "bZzyPscbtI8",
     "t": "Building AI Agents in Pure Python - Beginner Course",
     "c": "Dave Ebbelaar",
     "v": 443848,
     "m": 46,
     "d": "2025-02",
     "core": true
    },
    {
     "id": "sal78ACtGTc",
     "t": "What's next for AI agentic workflows ft. Andrew Ng of AI Fund",
     "c": "Sequoia Capital",
     "v": 411150,
     "m": 13,
     "d": "2024-03",
     "core": true
    },
    {
     "id": "ZVPlLaehjLk",
     "t": "Agentic AI Frameworks Explained: Workflows, Multi-Agent, & Production",
     "c": "IBM Technology",
     "v": 41110,
     "m": 11,
     "d": "2026-07",
     "core": false
    }
   ],
   "lab": {
    "t": "Write an agent loop from scratch",
    "g": "Implement thought, action and observation in about a hundred lines, with no framework."
   },
   "mini": {
    "t": "Same task, two designs",
    "g": "Solve one task as a fixed workflow and as an agent, then argue which was right."
   },
   "widgets": [
    "agentloop"
   ]
  },
  {
   "n": 13,
   "id": "m13",
   "file": "m13-mcp.html",
   "phase": 4,
   "title": "MCP - giving any AI safe access to your tools",
   "promise": "Build real MCP servers with the 2026 SDK, and prove they work without depending on any vendor's app.",
   "hours": 7,
   "concepts": [
    "Model Context Protocol",
    "tools resources prompts",
    "stdio transport",
    "streamable HTTP",
    "JSON-RPC",
    "stateless core",
    "MCP client",
    "tool poisoning",
    "MCP hosts"
   ],
   "videos": [
    {
     "id": "7j_NE6Pjv-E",
     "t": "Model Context Protocol (MCP), clearly explained (why it matters)",
     "c": "Greg Isenberg",
     "v": 1319813,
     "m": 20,
     "d": "2025-03",
     "core": true
    },
    {
     "id": "5xqFjh56AwM",
     "t": "MCP Crash Course: What Python Developers Need to Know",
     "c": "Dave Ebbelaar",
     "v": 257797,
     "m": 57,
     "d": "2025-04",
     "core": true
    },
    {
     "id": "kOhLoixrJXo",
     "t": "MCP In 26 Minutes (Model Context Protocol)",
     "c": "Tina Huang",
     "v": 240059,
     "m": 26,
     "d": "2025-10",
     "core": true
    },
    {
     "id": "cGuyrANVi4A",
     "t": "How Model Context Protocol (MCP) actually works",
     "c": "Google Cloud Tech",
     "v": 250953,
     "m": 7,
     "d": "2026-06",
     "core": false
    },
    {
     "id": "GAyNvq6Ayps",
     "t": "Running LLMs Locally Just Got Way Better - Ollama + MCP",
     "c": "Tech With Tim",
     "v": 152405,
     "m": 21,
     "d": "2026-04",
     "core": false
    },
    {
     "id": "DosHnyq78xY",
     "t": "Intro to MCP Servers – Model Context Protocol with Python Course",
     "c": "freeCodeCamp.org",
     "v": 103319,
     "m": 72,
     "d": "2025-10",
     "core": false
    },
    {
     "id": "Y0tZ35dFFx4",
     "t": "10 - MCP Inspector: Test and Debug your MCP Server Locally",
     "c": "AI Anytime",
     "v": 23218,
     "m": 8,
     "d": "2025-04",
     "core": false
    },
    {
     "id": "N3vHJcHBS-w",
     "t": "Model Context Protocol (MCP) Explained in 20 Minutes",
     "c": "Shaw Talebi",
     "v": 217668,
     "m": 19,
     "d": "2025-04",
     "core": false
    },
    {
     "id": "tlAxnGTkBEg",
     "t": "Breaking Down The NEW Stateless MCP Spec",
     "c": "Agentic Intelligence w/ Michael Levan",
     "v": 54,
     "m": 7,
     "d": "2026-08",
     "core": false
    }
   ],
   "lab": {
    "t": "Build three MCP servers",
    "g": "Write MCP servers with the v2 SDK and prove each one works using a client you wrote yourself."
   },
   "mini": {
    "t": "An MCP server for your own life",
    "g": "Build a server exposing something you personally use, and drive it from your own client."
   },
   "widgets": [
    "mcpwire"
   ]
  },
  {
   "n": 14,
   "id": "m14",
   "file": "m14-agent-frameworks.html",
   "phase": 4,
   "title": "Agent frameworks, and how to choose",
   "promise": "Run the real frameworks against the same task, and learn to pick one for reasons you can defend.",
   "hours": 5,
   "concepts": [
    "LangGraph",
    "CrewAI",
    "OpenAI Agents SDK",
    "Claude Agent SDK",
    "Google ADK",
    "Microsoft Agent Framework",
    "AWS Strands",
    "state and checkpointing",
    "choosing a framework"
   ],
   "videos": [
    {
     "id": "TD2ihEBkdkY",
     "t": "Building Intelligent Agents with Strands: A Hands-On Guide",
     "c": "AWS Developers",
     "v": 241529,
     "m": 10,
     "d": "2025-06",
     "core": true
    },
    {
     "id": "LSk5KaEGVk4",
     "t": "Agentic AI Engineering: Complete 4-Hour Workshop feat. MCP, CrewAI and OpenAI Agents SDK",
     "c": "Jon Krohn",
     "v": 222565,
     "m": 214,
     "d": "2025-06",
     "core": true
    },
    {
     "id": "ChaQ_tZDBFg",
     "t": "You Can Build The Craziest Things with Claudes Agent SDK",
     "c": "Traversy Media",
     "v": 51539,
     "m": 14,
     "d": "2026-03",
     "core": true
    },
    {
     "id": "JngWQLylFX8",
     "t": "Getting Started with Microsoft Agent Framework: Build Practical AI Agents",
     "c": "Microsoft Reactor",
     "v": 6929,
     "m": 67,
     "d": "2026-05",
     "core": false
    },
    {
     "id": "0Z0GUDakR_A",
     "t": "Intro to multi-agent systems with ADK",
     "c": "Google Cloud Tech",
     "v": 16775,
     "m": 11,
     "d": "2026-06",
     "core": false
    }
   ],
   "lab": {
    "t": "Rebuild your agent in a framework",
    "g": "Port your hand-written agent to LangGraph and run both against the same tasks."
   },
   "mini": {
    "t": "Pick and defend",
    "g": "Choose a framework for a stated scenario and write the three-sentence justification."
   },
   "widgets": [
    "chooser"
   ]
  },
  {
   "n": 15,
   "id": "m15",
   "file": "m15-multi-agent-memory.html",
   "phase": 4,
   "title": "Many agents, memory, and A2A",
   "promise": "Make several agents work together without them quietly failing, and give them memory that survives a restart.",
   "hours": 5,
   "concepts": [
    "multi-agent systems",
    "supervisor pattern",
    "handoffs",
    "short-term memory",
    "long-term memory",
    "A2A protocol",
    "agent cards",
    "MAST failure modes",
    "context isolation"
   ],
   "videos": [
    {
     "id": "Fbr_Solax1w",
     "t": "Introduction to Agent2Agent (A2A) Protocol",
     "c": "Google Cloud Tech",
     "v": 87753,
     "m": 7,
     "d": "2026-02",
     "core": true
    },
    {
     "id": "BM39OouLNsM",
     "t": "Build an End-to-End Multi-Agent AI System with LangGraph, MCP, Supervisor, Guardrails Safety & HITL",
     "c": "DSwithBappy",
     "v": 12245,
     "m": 314,
     "d": "2026-07",
     "core": true
    }
   ],
   "lab": {
    "t": "Two agents, one job",
    "g": "Build a researcher and a writer that share memory and hand work to each other."
   },
   "mini": {
    "t": "Make it fail on purpose",
    "g": "Cause one named MAST failure mode in your pair, then fix it and prove the fix."
   },
   "widgets": []
  },
  {
   "n": 16,
   "id": "m16",
   "file": "m16-evaluation-observability.html",
   "phase": 5,
   "title": "Knowing whether your agent works",
   "promise": "Trace what your agent actually did, and build evaluations that catch problems before your users find them.",
   "hours": 6,
   "concepts": [
    "tracing",
    "spans",
    "OpenTelemetry GenAI",
    "LLM as a judge",
    "error analysis",
    "binary evaluation",
    "TPR and TNR",
    "Cohen's kappa",
    "regression testing",
    "Phoenix",
    "Langfuse"
   ],
   "videos": [
    {
     "id": "d5EltXhbcfA",
     "t": "Building and evaluating AI Agents — Sayash Kapoor, AI Snake Oil",
     "c": "AI Engineer",
     "v": 239553,
     "m": 19,
     "d": "2025-04",
     "core": true
    },
    {
     "id": "BsWxPI9UM4c",
     "t": "Why AI evals are the hottest new skill for product builders | Hamel Husain & Shreya Shankar",
     "c": "Lenny's Podcast",
     "v": 129115,
     "m": 106,
     "d": "2025-09",
     "core": true
    },
    {
     "id": "a3SMraZWNNs",
     "t": "How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)",
     "c": "Dave Ebbelaar",
     "v": 82253,
     "m": 55,
     "d": "2025-09",
     "core": true
    },
    {
     "id": "pnlT_xatpVQ",
     "t": "LLM-as-a-Judge 101",
     "c": "Arize AI",
     "v": 3366,
     "m": 43,
     "d": "2025-11",
     "core": false
    },
    {
     "id": "wcYnzHJlUR0",
     "t": "LLM Eval Tools Compared: Arize Phoenix",
     "c": "Hamel Husain",
     "v": 3054,
     "m": 32,
     "d": "2025-10",
     "core": false
    },
    {
     "id": "wQ5ivl44mQw",
     "t": "Observing and Improving AI Agents with Langfuse",
     "c": "ClickHouse",
     "v": 446,
     "m": 58,
     "d": "2026-07",
     "core": false
    }
   ],
   "lab": {
    "t": "Trace and judge your agent",
    "g": "Instrument your agent with Phoenix, then build a binary judge and calibrate it against your own labels."
   },
   "mini": {
    "t": "Catch a real regression",
    "g": "Change your prompt to make quality worse and confirm your evals catch it before you would have."
   },
   "widgets": []
  },
  {
   "n": 17,
   "id": "m17",
   "file": "m17-deploy-cost-safety.html",
   "phase": 5,
   "title": "Shipping safely and cheaply",
   "promise": "Harden your agent against attacks that really happen, put a ceiling on what it can cost, and run it somewhere other than your laptop.",
   "hours": 6,
   "concepts": [
    "prompt injection",
    "lethal trifecta",
    "OWASP agentic top 10",
    "guardrails",
    "sandboxing",
    "least privilege",
    "cost control",
    "FinOps",
    "deployment options"
   ],
   "videos": [
    {
     "id": "Qvx2sVgQ-u0",
     "t": "Hacking AI is TOO EASY (this should be illegal)",
     "c": "NetworkChuck",
     "v": 1273294,
     "m": 26,
     "d": "2025-08",
     "core": true
    },
    {
     "id": "rAEqP9VEhe8",
     "t": "Generative AI's Greatest Flaw - Computerphile",
     "c": "Computerphile",
     "v": 604740,
     "m": 12,
     "d": "2025-02",
     "core": true
    },
    {
     "id": "gUNXZMcd2jU",
     "t": "OWASP's Top 10 Ways to Attack LLMs: AI Vulnerabilities Exposed",
     "c": "IBM Technology",
     "v": 275228,
     "m": 25,
     "d": "2026-03",
     "core": true
    },
    {
     "id": "5ZA1lTxTH3c",
     "t": "Securing AI Agents: How to Prevent Hidden Prompt Injection Attacks",
     "c": "IBM Technology",
     "v": 27352,
     "m": 10,
     "d": "2026-01",
     "core": false
    },
    {
     "id": "7Z7ID5BbZU4",
     "t": "Containers Don't Make Your AI Agent Safe",
     "c": "Web Dev Simplified",
     "v": 38815,
     "m": 36,
     "d": "2026-06",
     "core": false
    },
    {
     "id": "cNE7P5FkqR8",
     "t": "Andrew Bullen - Breaking the Lethal Trifecta (Without Ruining Your Agents) | [un]prompted 2026",
     "c": "unprompted",
     "v": 673,
     "m": 19,
     "d": "2026-03",
     "core": false
    }
   ],
   "lab": {
    "t": "Attack and harden your own agent",
    "g": "Break your agent with prompt injection, then close the hole in code rather than in the prompt."
   },
   "mini": {
    "t": "Give it a budget it cannot exceed",
    "g": "Add a hard cost ceiling that stops the agent mid-run, and prove it triggers."
   },
   "widgets": [
    "injection",
    "cost"
   ]
  },
  {
   "n": 18,
   "id": "m18",
   "file": "m18-capstone.html",
   "phase": 6,
   "title": "Capstone - your research and document agent",
   "promise": "Combine everything into one agent that researches, reads your documents, cites its sources, and is measured and traced.",
   "hours": 12,
   "concepts": [
    "system design",
    "requirements",
    "citations",
    "evaluation set",
    "tracing in production",
    "cost ceiling",
    "shipping",
    "documentation"
   ],
   "videos": [
    {
     "id": "E8zpgNPx8jE",
     "t": "Build a Complete End-to-End GenAI Project in 3 Hours",
     "c": "Dave Ebbelaar",
     "v": 139262,
     "m": 178,
     "d": "2025-11",
     "core": true
    },
    {
     "id": "LSk5KaEGVk4",
     "t": "Agentic AI Engineering: Complete 4-Hour Workshop feat. MCP, CrewAI and OpenAI Agents SDK",
     "c": "Jon Krohn",
     "v": 222565,
     "m": 214,
     "d": "2025-06",
     "core": false
    },
    {
     "id": "BM39OouLNsM",
     "t": "Build an End-to-End Multi-Agent AI System with LangGraph, MCP, Supervisor, Guardrails Safety & HITL",
     "c": "DSwithBappy",
     "v": 12245,
     "m": 314,
     "d": "2026-07",
     "core": false
    }
   ],
   "lab": {
    "t": "Build and ship the capstone",
    "g": "Assemble the research and document agent end to end, evaluate it, and deploy it."
   },
   "mini": {
    "t": "Ship it",
    "g": "Deploy your agent somewhere reachable and write the README a stranger could follow."
   },
   "widgets": []
  },
  {
   "n": 19,
   "id": "m19",
   "file": "m19-tokenizer.html",
   "phase": 7,
   "title": "B1 · Train your own tokenizer",
   "promise": "Build byte pair encoding from scratch and train it on your own text, so you own the thing that decides what a token is.",
   "hours": 3,
   "concepts": [
    "byte pair encoding",
    "merge table",
    "vocabulary size",
    "compression ratio",
    "lossless round trip",
    "training a tokenizer",
    "why chunks not words"
   ],
   "videos": [
    {
     "id": "zduSFxRajkE",
     "t": "Let's build the GPT Tokenizer",
     "c": "Andrej Karpathy",
     "v": 1161030,
     "m": 133,
     "d": "2024-02",
     "core": true
    },
    {
     "id": "nKSk_TiR8YA",
     "t": "Most devs don't understand how LLM tokens work",
     "c": "Matt Pocock",
     "v": 330146,
     "m": 10,
     "d": "2025-09",
     "core": true
    }
   ],
   "lab": {
    "t": "Train a tokenizer",
    "g": "Implement BPE and train it on a corpus."
   },
   "mini": {
    "t": "Your own tokenizer",
    "g": "Train on your own text and check it is lossless."
   },
   "widgets": []
  },
  {
   "n": 20,
   "id": "m20",
   "file": "m20-transformer.html",
   "phase": 7,
   "title": "B2 · Build the transformer",
   "promise": "Write attention, heads and blocks yourself in about 150 lines, then prove the causal mask works.",
   "hours": 4,
   "concepts": [
    "query key value",
    "scaled dot product",
    "causal mask",
    "multi-head attention",
    "residual connections",
    "layer norm",
    "position embedding",
    "parameter count"
   ],
   "videos": [
    {
     "id": "eMlx5fFNoYc",
     "t": "Attention in transformers, step-by-step | Deep Learning Chapter 6",
     "c": "3Blue1Brown",
     "v": 4369296,
     "m": 26,
     "d": "2024-04",
     "core": true
    },
    {
     "id": "wjZofJX0v4M",
     "t": "Transformers, the tech behind LLMs | Deep Learning Chapter 5",
     "c": "3Blue1Brown",
     "v": 10837072,
     "m": 27,
     "d": "2024-04",
     "core": true
    },
    {
     "id": "zxQyTK8quyY",
     "t": "Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!",
     "c": "StatQuest with Josh Starmer",
     "v": 1161820,
     "m": 36,
     "d": "2023-07",
     "core": false
    }
   ],
   "lab": {
    "t": "Build TinyGPT",
    "g": "Write the model, check the shapes."
   },
   "mini": {
    "t": "Prove the mask",
    "g": "Show it cannot see the future, without training it."
   },
   "widgets": [
    "modelsize",
    "attention"
   ]
  },
  {
   "n": 21,
   "id": "m21",
   "file": "m21-train-gpt.html",
   "phase": 7,
   "title": "B3 · Train your own GPT",
   "promise": "Take the model you wrote and train it from random numbers until it writes readable English.",
   "hours": 5,
   "concepts": [
    "next token prediction",
    "loss and the random baseline",
    "validation split",
    "batch and context",
    "learning rate",
    "training curve",
    "sampling",
    "temperature"
   ],
   "videos": [
    {
     "id": "kCc8FmEb1nY",
     "t": "Let's build GPT: from scratch, in code, spelled out.",
     "c": "Andrej Karpathy",
     "v": 7609153,
     "m": 116,
     "d": "2023-01",
     "core": true
    },
    {
     "id": "VMj-3S1tku0",
     "t": "The spelled-out intro to neural networks and backpropagation: building micrograd",
     "c": "Andrej Karpathy",
     "v": 3878022,
     "m": 145,
     "d": "2022-08",
     "core": true
    }
   ],
   "lab": {
    "t": "Train it",
    "g": "Train on a corpus and watch the loss fall."
   },
   "mini": {
    "t": "Train on your own text",
    "g": "Beat the random baseline and prove it."
   },
   "widgets": [
    "modelsize"
   ]
  },
  {
   "n": 22,
   "id": "m22",
   "file": "m22-pretraining.html",
   "phase": 7,
   "title": "B4 · Why pretraining matters",
   "promise": "Your model writes English-shaped nonsense. Fine-tune a pretrained one in two minutes and see exactly what pretraining bought.",
   "hours": 4,
   "concepts": [
    "pretraining",
    "scale",
    "fine-tuning",
    "LoRA",
    "instruction following",
    "when from scratch is right",
    "the honest comparison"
   ],
   "videos": [
    {
     "id": "7xTGNNLPyMI",
     "t": "Deep Dive into LLMs like ChatGPT",
     "c": "Andrej Karpathy",
     "v": 8386781,
     "m": 211,
     "d": "2025-02",
     "core": true
    },
    {
     "id": "t1caDsMzWBk",
     "t": "LoRA & QLoRA Fine-tuning Explained In-Depth",
     "c": "Mark Hennings",
     "v": 181296,
     "m": 14,
     "d": "2023-12",
     "core": true
    },
    {
     "id": "CcrC5zSv1iA",
     "t": "LLM Fine-Tuning Course – From Supervised FT to RLHF, LoRA, and Multimodal",
     "c": "freeCodeCamp.org",
     "v": 85990,
     "m": 716,
     "d": "2026-03",
     "core": false
    }
   ],
   "lab": {
    "t": "Fine-tune a pretrained model",
    "g": "LoRA on a small model, on your processor."
   },
   "mini": {
    "t": "Compare them",
    "g": "Same prompts, both models, written verdict."
   },
   "widgets": []
  },
  {
   "n": 23,
   "id": "m23",
   "file": "m23-serve-your-model.html",
   "phase": 7,
   "title": "B5 · Serve it, and plug it in",
   "promise": "Put your model behind the same interface every provider uses, then point the whole course at it.",
   "hours": 4,
   "concepts": [
    "OpenAI-compatible endpoint",
    "chat completions shape",
    "serving a model",
    "token counting",
    "LLM_PROVIDER",
    "where a tiny model belongs"
   ],
   "videos": [
    {
     "id": "GAyNvq6Ayps",
     "t": "Running LLMs Locally Just Got Way Better - Ollama + MCP",
     "c": "Tech With Tim",
     "v": 152405,
     "m": 21,
     "d": "2026-04",
     "core": true
    },
    {
     "id": "pTaSDVz0gok",
     "t": "EASIEST Way to Fine-Tune a LLM and Use It With Ollama",
     "c": "Tech With Tim",
     "v": 345342,
     "m": 22,
     "d": "2025-06",
     "core": true
    }
   ],
   "lab": {
    "t": "Serve your model",
    "g": "A standard-library server for your own weights."
   },
   "mini": {
    "t": "Run the course on it",
    "g": "Call it through llm.py and record what happens."
   },
   "widgets": []
  }
 ]
};
